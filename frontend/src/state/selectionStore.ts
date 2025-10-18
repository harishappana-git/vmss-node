import { create } from 'zustand'

type Selection = {
  kind: 'node' | 'gpu' | 'link' | 'kernel'
  id: string
}

type SelectionState = {
  selected: Selection | null
  select: (selection: Selection | null) => void
}

export const useSelectionStore = create<SelectionState>((set) => ({
  selected: null,
  select: (selection) => set({ selected: selection })
}))
