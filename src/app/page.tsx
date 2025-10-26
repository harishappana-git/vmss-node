import dynamic from "next/dynamic";

const Visualizer = dynamic(() => import("./Visualizer"), { ssr: false });

export default function Page() {
  return <Visualizer />;
}
