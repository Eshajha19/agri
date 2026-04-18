import { Loader2 } from "lucide-react";



export default function Loader({ fullScreen = false, size = 24 }) {
  const spinner = (
    <Loader2
  style={{
    width: size,
    height: size,
    animation: "spin 0.8s linear infinite"
  }}
/>
  );

  if (!fullScreen) return spinner;

  return (
    <div
      className="fixed inset-0 flex items-center justify-center bg-white/60 z-50"
      aria-busy="true"
    >
      {spinner}
    </div>
  );
}