package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/example/vmss-node/backend/internal/data"
	"github.com/example/vmss-node/backend/internal/metrics"
	"github.com/example/vmss-node/backend/internal/server"
)

func main() {
	topology := data.DemoTopology()
	kernels := data.DemoKernels()
	aggregator := metrics.NewAggregator(topology, kernels)

	server := server.New(aggregator)
	httpServer := &http.Server{
		Addr:    ":8080",
		Handler: server.Routes(),
	}

	stop := make(chan struct{})
	go aggregator.UpdateLoop(stop)

	go func() {
		log.Printf("server listening on %s", httpServer.Addr)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("http server: %v", err)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	<-sigCh
	close(stop)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(ctx); err != nil {
		log.Printf("shutdown error: %v", err)
	}
}
