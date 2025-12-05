package data

import "time"

type GPU struct {
	UUID        string    `json:"uuid"`
	Model       string    `json:"model"`
	Mig         MIG       `json:"mig"`
	NvlinkPeers []string  `json:"nvlinkPeers"`
	LastUpdated time.Time `json:"lastUpdated"`
}

type MIG struct {
	Enabled   bool     `json:"enabled"`
	Instances []string `json:"instances"`
}

type Node struct {
	ID       string `json:"id"`
	Hostname string `json:"hostname"`
	Gpus     []GPU  `json:"gpus"`
	IB       IBPort `json:"ib"`
}

type IBPort struct {
	PortGUID    string  `json:"portGuid"`
	SpeedGbps   float64 `json:"speedGbps"`
	Health      string  `json:"health"`
	Utilization float64 `json:"utilization"`
}

type LinkType string

const (
	LinkIB     LinkType = "IB"
	LinkNVLink LinkType = "NVLINK"
)

type Link struct {
	ID           string   `json:"id"`
	Type         LinkType `json:"type"`
	From         string   `json:"from"`
	To           string   `json:"to"`
	CapacityGbps float64  `json:"capacityGbps"`
}

type Rack struct {
	ID    string `json:"id"`
	Nodes []Node `json:"nodes"`
}

type Cluster struct {
	ID    string `json:"id"`
	Racks []Rack `json:"racks"`
	Links []Link `json:"links"`
}

type Topology struct {
	RegionID string    `json:"regionId"`
	Clusters []Cluster `json:"clusters"`
}

type Kernel struct {
	ID        string  `json:"id"`
	Name      string  `json:"name"`
	GridDim   int     `json:"gridDim"`
	BlockDim  int     `json:"blockDim"`
	Occupancy float64 `json:"occupancy"`
	SmEff     float64 `json:"smEff"`
	DramBwGBs float64 `json:"dramBwGBs"`
}

type KernelBlockMapping struct {
	BlockID int    `json:"blockId"`
	GPUUUID string `json:"gpuUuid"`
	SM      int    `json:"sm"`
}

// Metrics frames

type GPUFrame struct {
	Timestamp  time.Time `json:"t"`
	Topic      string    `json:"topic"`
	Util       float64   `json:"util"`
	MemUsedGB  float64   `json:"memUsedGB"`
	NvlinkGBs  float64   `json:"nvlinkGBs"`
	PCIETxGBs  float64   `json:"pcieTxGBs"`
	TempC      float64   `json:"tempC"`
	PowerW     float64   `json:"powerW"`
	EccCorrect int       `json:"eccCorrected"`
	EccUncor   int       `json:"eccUncorrected"`
}

type LinkFrame struct {
	Timestamp time.Time `json:"t"`
	LinkID    string    `json:"linkId"`
	BwGbps    float64   `json:"bwGbps"`
	RttUs     float64   `json:"rttUs"`
	Errors    float64   `json:"errs"`
}

type NodeFrame struct {
	Timestamp   time.Time `json:"t"`
	NodeID      string    `json:"nodeId"`
	CPUUtil     float64   `json:"cpuUtil"`
	MemoryUsed  float64   `json:"memoryUsedGB"`
	IBUtilGbps  float64   `json:"ibUtilGbps"`
	JobsRunning int       `json:"jobsRunning"`
}

type KernelFrame struct {
	Timestamp time.Time `json:"t"`
	KernelID  string    `json:"kernelId"`
	Occupancy float64   `json:"occupancy"`
	SmEff     float64   `json:"smEff"`
	DramBW    float64   `json:"dramBwGBs"`
	Active    bool      `json:"active"`
}

// Historical buffers keep a rolling window of frames.
type HistoryBuffer[T any] struct {
	Frames []T
	Limit  int
}

func NewHistoryBuffer[T any](limit int) HistoryBuffer[T] {
	return HistoryBuffer[T]{Frames: make([]T, 0, limit), Limit: limit}
}

func (h *HistoryBuffer[T]) Add(frame T) {
	if len(h.Frames) == h.Limit {
		copy(h.Frames, h.Frames[1:])
		h.Frames[len(h.Frames)-1] = frame
		return
	}
	h.Frames = append(h.Frames, frame)
}

func (h *HistoryBuffer[T]) Snapshot() []T {
	frames := make([]T, len(h.Frames))
	copy(frames, h.Frames)
	return frames
}
