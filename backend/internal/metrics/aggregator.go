package metrics

import (
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/example/vmss-node/backend/internal/data"
)

type Aggregator struct {
	topology data.Topology
	kernels  []data.Kernel

	gpuFrames   map[string]data.HistoryBuffer[data.GPUFrame]
	nodeFrames  map[string]data.HistoryBuffer[data.NodeFrame]
	linkFrames  map[string]data.HistoryBuffer[data.LinkFrame]
	kernelState map[string]data.HistoryBuffer[data.KernelFrame]

	mutex sync.RWMutex
}

func NewAggregator(topology data.Topology, kernels []data.Kernel) *Aggregator {
	agg := &Aggregator{
		topology:    topology,
		kernels:     kernels,
		gpuFrames:   make(map[string]data.HistoryBuffer[data.GPUFrame]),
		nodeFrames:  make(map[string]data.HistoryBuffer[data.NodeFrame]),
		linkFrames:  make(map[string]data.HistoryBuffer[data.LinkFrame]),
		kernelState: make(map[string]data.HistoryBuffer[data.KernelFrame]),
	}

	agg.seedHistory()
	return agg
}

func (a *Aggregator) seedHistory() {
	now := time.Now()
	for _, cluster := range a.topology.Clusters {
		for _, rack := range cluster.Racks {
			for _, node := range rack.Nodes {
				buf := data.NewHistoryBuffer[data.NodeFrame](3600)
				frame := data.NodeFrame{
					Timestamp:   now,
					NodeID:      node.ID,
					CPUUtil:     0.45,
					MemoryUsed:  180.0,
					IBUtilGbps:  80,
					JobsRunning: 4,
				}
				buf.Add(frame)
				a.nodeFrames[node.ID] = buf

				for _, gpu := range node.Gpus {
					gBuf := data.NewHistoryBuffer[data.GPUFrame](3600)
					gFrame := data.GPUFrame{
						Timestamp:  now,
						Topic:      "gpu." + gpu.UUID,
						Util:       0.6,
						MemUsedGB:  50,
						NvlinkGBs:  150,
						PCIETxGBs:  12,
						TempC:      65,
						PowerW:     280,
						EccCorrect: 0,
						EccUncor:   0,
					}
					gBuf.Add(gFrame)
					a.gpuFrames[gpu.UUID] = gBuf
				}
			}
		}
		for _, link := range cluster.Links {
			lBuf := data.NewHistoryBuffer[data.LinkFrame](720)
			frame := data.LinkFrame{
				Timestamp: now,
				LinkID:    link.ID,
				BwGbps:    link.CapacityGbps * 0.4,
				RttUs:     8.2,
				Errors:    0,
			}
			lBuf.Add(frame)
			a.linkFrames[link.ID] = lBuf
		}
	}

	for _, kernel := range a.kernels {
		nowFrame := data.NewHistoryBuffer[data.KernelFrame](3600)
		frame := data.KernelFrame{
			Timestamp: now,
			KernelID:  kernel.ID,
			Occupancy: kernel.Occupancy,
			SmEff:     kernel.SmEff,
			DramBW:    kernel.DramBwGBs,
			Active:    true,
		}
		nowFrame.Add(frame)
		a.kernelState[kernel.ID] = nowFrame
	}
}

func (a *Aggregator) Topology() data.Topology {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	return a.topology
}

func (a *Aggregator) Kernels() []data.Kernel {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	kernels := make([]data.Kernel, len(a.kernels))
	copy(kernels, a.kernels)
	return kernels
}

func (a *Aggregator) NodeHistory(id string) []data.NodeFrame {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	if buf, ok := a.nodeFrames[id]; ok {
		return buf.Snapshot()
	}
	return nil
}

func (a *Aggregator) GPUHistory(uuid string) []data.GPUFrame {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	if buf, ok := a.gpuFrames[uuid]; ok {
		return buf.Snapshot()
	}
	return nil
}

func (a *Aggregator) LinkHistory(id string) []data.LinkFrame {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	if buf, ok := a.linkFrames[id]; ok {
		return buf.Snapshot()
	}
	return nil
}

func (a *Aggregator) KernelHistory(id string) []data.KernelFrame {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	if buf, ok := a.kernelState[id]; ok {
		return buf.Snapshot()
	}
	return nil
}

func (a *Aggregator) UpdateLoop(stop <-chan struct{}) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			a.step(time.Now())
		case <-stop:
			return
		}
	}
}

func (a *Aggregator) step(ts time.Time) {
	a.mutex.Lock()
	defer a.mutex.Unlock()

	rand.Seed(ts.UnixNano())
	for id, buf := range a.nodeFrames {
		last := buf.Frames[len(buf.Frames)-1]
		frame := last
		frame.Timestamp = ts
		frame.CPUUtil = clamp01(jitter(last.CPUUtil, 0.05))
		frame.MemoryUsed = math.Max(0, last.MemoryUsed+jitter(0, 5))
		frame.IBUtilGbps = math.Max(0, last.IBUtilGbps+jitter(0, 10))
		frame.JobsRunning = max(0, last.JobsRunning+rand.Intn(2)-1)
		buf.Add(frame)
		a.nodeFrames[id] = buf
	}

	for id, buf := range a.gpuFrames {
		last := buf.Frames[len(buf.Frames)-1]
		frame := last
		frame.Timestamp = ts
		frame.Util = clamp01(jitter(last.Util, 0.08))
		frame.MemUsedGB = math.Max(0, last.MemUsedGB+jitter(0, 2))
		frame.NvlinkGBs = math.Max(0, last.NvlinkGBs+jitter(0, 10))
		frame.PCIETxGBs = math.Max(0, last.PCIETxGBs+jitter(0, 1))
		frame.TempC = math.Max(20, last.TempC+jitter(0, 1.5))
		frame.PowerW = math.Max(150, last.PowerW+jitter(0, 8))
		buf.Add(frame)
		a.gpuFrames[id] = buf
	}

	for id, buf := range a.linkFrames {
		last := buf.Frames[len(buf.Frames)-1]
		frame := last
		frame.Timestamp = ts
		frame.BwGbps = math.Max(0, last.BwGbps+jitter(0, 15))
		frame.RttUs = math.Max(4, last.RttUs+jitter(0, 0.5))
		buf.Add(frame)
		a.linkFrames[id] = buf
	}

	for id, buf := range a.kernelState {
		last := buf.Frames[len(buf.Frames)-1]
		frame := last
		frame.Timestamp = ts
		frame.Occupancy = clamp01(jitter(last.Occupancy, 0.03))
		frame.SmEff = clamp01(jitter(last.SmEff, 0.02))
		frame.DramBW = math.Max(0, last.DramBW+jitter(0, 0.3))
		buf.Add(frame)
		a.kernelState[id] = buf
	}
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func jitter(base float64, delta float64) float64 {
	return base + (rand.Float64()*2-1)*delta
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (a *Aggregator) UpdateTopology(topology data.Topology) {
	a.mutex.Lock()
	defer a.mutex.Unlock()
	a.topology = topology
}

func (a *Aggregator) UpdateKernels(kernels []data.Kernel) {
	a.mutex.Lock()
	defer a.mutex.Unlock()
	a.kernels = kernels
}
