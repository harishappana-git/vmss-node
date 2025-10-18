package data

import (
	"fmt"
)

func DemoTopology() Topology {
	region := Topology{RegionID: "lab"}
	for c := 0; c < 2; c++ {
		clusterID := fmt.Sprintf("c%d", c)
		cluster := Cluster{ID: clusterID}
		for r := 0; r < 2; r++ {
			rackID := fmt.Sprintf("%s-r%d", clusterID, r)
			rack := Rack{ID: rackID}
			for n := 0; n < 4; n++ {
				nodeID := fmt.Sprintf("%s-n%d", clusterID, r*4+n)
				node := Node{
					ID:       nodeID,
					Hostname: fmt.Sprintf("%s-host-%d", clusterID, r*4+n),
					IB: IBPort{
						PortGUID:    fmt.Sprintf("0x%04x", 1000+c*100+r*10+n),
						SpeedGbps:   400,
						Health:      "healthy",
						Utilization: 0.4,
					},
				}
				for g := 0; g < 8; g++ {
					uuid := fmt.Sprintf("GPU-%s-%d", nodeID, g)
					gpu := GPU{
						UUID:  uuid,
						Model: "H100-SXM-80G",
						Mig:   MIG{Enabled: false, Instances: []string{}},
					}
					node.Gpus = append(node.Gpus, gpu)
				}
				rack.Nodes = append(rack.Nodes, node)
			}
			cluster.Racks = append(cluster.Racks, rack)
		}
		for i := 0; i < 8; i++ {
			from := fmt.Sprintf("%s-n%d:ib0", clusterID, i)
			to := fmt.Sprintf("%s-n%d:ib0", clusterID, (i+1)%8)
			link := Link{
				ID:           fmt.Sprintf("%s-link-%d", clusterID, i),
				Type:         LinkIB,
				From:         from,
				To:           to,
				CapacityGbps: 400,
			}
			cluster.Links = append(cluster.Links, link)
		}
		region.Clusters = append(region.Clusters, cluster)
	}
	return region
}

func DemoKernels() []Kernel {
	return []Kernel{
		{ID: "k-gemm", Name: "gemm_fp16", GridDim: 4096, BlockDim: 256, Occupancy: 0.82, SmEff: 0.87, DramBwGBs: 1.9},
		{ID: "k-attn", Name: "scaled_dot_attn", GridDim: 8192, BlockDim: 256, Occupancy: 0.85, SmEff: 0.9, DramBwGBs: 2.1},
	}
}
