package server

import (
	"bufio"
	"crypto/sha1"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/example/vmss-node/backend/internal/data"
	"github.com/example/vmss-node/backend/internal/metrics"
)

type Server struct {
	aggregator *metrics.Aggregator
}

func New(aggregator *metrics.Aggregator) *Server {
	return &Server{aggregator: aggregator}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	mux.HandleFunc("/v1/topology", s.handleTopology)
	mux.HandleFunc("/v1/kernels", s.handleKernels)
	mux.HandleFunc("/v1/search", s.handleSearch)
	mux.HandleFunc("/v1/node/", s.handleNode)
	mux.HandleFunc("/v1/gpu/", s.handleGPU)
	mux.HandleFunc("/v1/link/", s.handleLink)
	mux.HandleFunc("/stream", s.handleStream)
	return s.withJSONHeaders(mux)
}

func (s *Server) withJSONHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		next.ServeHTTP(w, r)
	})
}

func (s *Server) handleTopology(w http.ResponseWriter, r *http.Request) {
	topology := s.aggregator.Topology()
	s.respondJSON(w, topology)
}

func (s *Server) handleKernels(w http.ResponseWriter, r *http.Request) {
	kernels := s.aggregator.Kernels()
	s.respondJSON(w, map[string]any{"kernels": kernels})
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	if query == "" {
		s.respondJSON(w, map[string]any{"results": []any{}})
		return
	}
	query = strings.ToLower(query)

	topology := s.aggregator.Topology()
	var results []map[string]string
	for _, cluster := range topology.Clusters {
		for _, rack := range cluster.Racks {
			for _, node := range rack.Nodes {
				if strings.Contains(strings.ToLower(node.ID), query) || strings.Contains(strings.ToLower(node.Hostname), query) {
					results = append(results, map[string]string{
						"type": "node",
						"id":   node.ID,
					})
				}
				for _, gpu := range node.Gpus {
					if strings.Contains(strings.ToLower(gpu.UUID), query) {
						results = append(results, map[string]string{
							"type": "gpu",
							"id":   gpu.UUID,
						})
					}
				}
			}
		}
	}
	s.respondJSON(w, map[string]any{"results": results})
}

func (s *Server) handleNode(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/v1/node/")
	if id == "" {
		http.NotFound(w, r)
		return
	}
	topology := s.aggregator.Topology()
	var node *data.Node
	for _, cluster := range topology.Clusters {
		for _, rack := range cluster.Racks {
			for i := range rack.Nodes {
				if rack.Nodes[i].ID == id {
					node = &rack.Nodes[i]
					break
				}
			}
		}
	}
	if node == nil {
		http.NotFound(w, r)
		return
	}

	history := s.aggregator.NodeHistory(id)
	s.respondJSON(w, map[string]any{
		"node":    node,
		"history": history,
	})
}

func (s *Server) handleGPU(w http.ResponseWriter, r *http.Request) {
	uuid := strings.TrimPrefix(r.URL.Path, "/v1/gpu/")
	if uuid == "" {
		http.NotFound(w, r)
		return
	}
	topology := s.aggregator.Topology()
	var gpu *data.GPU
	for _, cluster := range topology.Clusters {
		for _, rack := range cluster.Racks {
			for _, node := range rack.Nodes {
				for i := range node.Gpus {
					if node.Gpus[i].UUID == uuid {
						gpu = &node.Gpus[i]
						break
					}
				}
			}
		}
	}
	if gpu == nil {
		http.NotFound(w, r)
		return
	}

	history := s.aggregator.GPUHistory(uuid)
	s.respondJSON(w, map[string]any{
		"gpu":     gpu,
		"history": history,
	})
}

func (s *Server) handleLink(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/v1/link/")
	if id == "" {
		http.NotFound(w, r)
		return
	}
	history := s.aggregator.LinkHistory(id)
	if history == nil {
		http.NotFound(w, r)
		return
	}
	s.respondJSON(w, map[string]any{"history": history})
}

func (s *Server) handleStream(w http.ResponseWriter, r *http.Request) {
	if !strings.Contains(strings.ToLower(r.Header.Get("Connection")), "upgrade") || !strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
		http.Error(w, "upgrade required", http.StatusBadRequest)
		return
	}

	key := r.Header.Get("Sec-WebSocket-Key")
	if key == "" {
		http.Error(w, "missing websocket key", http.StatusBadRequest)
		return
	}

	hijacker, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "webserver doesn't support hijacking", http.StatusInternalServerError)
		return
	}

	conn, bufrw, err := hijacker.Hijack()
	if err != nil {
		log.Printf("hijack failed: %v", err)
		return
	}

	accept := computeAcceptKey(key)
	response := fmt.Sprintf("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n", accept)
	if _, err := bufrw.WriteString(response); err != nil {
		log.Printf("write handshake failed: %v", err)
		_ = conn.Close()
		return
	}
	if err := bufrw.Flush(); err != nil {
		log.Printf("flush handshake failed: %v", err)
		_ = conn.Close()
		return
	}

	defer conn.Close()

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case ts := <-ticker.C:
			if err := s.sendFrames(bufrw.Writer, ts); err != nil {
				log.Printf("send frame error: %v", err)
				return
			}
			if err := bufrw.Flush(); err != nil {
				log.Printf("flush frame error: %v", err)
				return
			}
		}
	}
}

func computeAcceptKey(key string) string {
	h := sha1.New()
	h.Write([]byte(key))
	h.Write([]byte("258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))
	return base64.StdEncoding.EncodeToString(h.Sum(nil))
}

func (s *Server) sendFrames(w *bufio.Writer, ts time.Time) error {
	topology := s.aggregator.Topology()

	type frameMsg struct {
		Topic string      `json:"topic"`
		Time  int64       `json:"t"`
		Data  interface{} `json:"data"`
	}

	var frames []frameMsg
	for _, cluster := range topology.Clusters {
		for _, rack := range cluster.Racks {
			for _, node := range rack.Nodes {
				if nodeHistory := s.aggregator.NodeHistory(node.ID); nodeHistory != nil {
					latest := nodeHistory[len(nodeHistory)-1]
					frames = append(frames, frameMsg{Topic: "node." + node.ID, Time: ts.UnixMilli(), Data: latest})
				}
				for _, gpu := range node.Gpus {
					if history := s.aggregator.GPUHistory(gpu.UUID); history != nil {
						latest := history[len(history)-1]
						frames = append(frames, frameMsg{Topic: "gpu." + gpu.UUID, Time: ts.UnixMilli(), Data: latest})
					}
				}
			}
		}
		for _, link := range cluster.Links {
			if history := s.aggregator.LinkHistory(link.ID); history != nil {
				latest := history[len(history)-1]
				frames = append(frames, frameMsg{Topic: "link." + link.ID, Time: ts.UnixMilli(), Data: latest})
			}
		}
	}

	for _, kernel := range s.aggregator.Kernels() {
		if history := s.aggregator.KernelHistory(kernel.ID); history != nil {
			latest := history[len(history)-1]
			frames = append(frames, frameMsg{Topic: "kernel." + kernel.ID, Time: ts.UnixMilli(), Data: latest})
		}
	}

	for _, frame := range frames {
		payload, err := json.Marshal(frame)
		if err != nil {
			return err
		}
		if err := writeWebSocketFrame(w, payload); err != nil {
			return err
		}
	}
	return nil
}

func writeWebSocketFrame(w *bufio.Writer, payload []byte) error {
	length := len(payload)
	var header []byte
	switch {
	case length <= 125:
		header = []byte{0x81, byte(length)}
	case length <= 65535:
		header = []byte{0x81, 126, byte(length >> 8), byte(length)}
	default:
		return fmt.Errorf("payload too large: %d", length)
	}
	if _, err := w.Write(header); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

func (s *Server) respondJSON(w http.ResponseWriter, payload any) {
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		log.Printf("encode error: %v", err)
	}
}
