import { useEffect, useRef, useState, useCallback } from "react";
import * as $3Dmol from "3dmol";
import { Maximize2, X, Loader2 } from "lucide-react";

export default function Molecule3DViewer({ sdfData }) {
  const viewerRef = useRef(null);
  const [isOpen, setIsOpen] = useState(false);
  const [isRendering, setIsRendering] = useState(true);

  /* ------------ Load molecule into compact viewer ------------ */
  const loadModel = useCallback(
    (target) => {
      if (!sdfData || !target) return;

      setIsRendering(true);

      setTimeout(() => {
        try {
          target.innerHTML = "";

          // FIX: Initialize without backgroundColor in config
          const viewer = $3Dmol.createViewer(target);

          // FIX: Set transparent background using the method (0x000000, 0)
          viewer.setBackgroundColor(0x000000, 1);

          viewer.addModel(sdfData, "sdf");
          viewer.setStyle(
            {},
            {
              stick: { radius: 0.15 },
              sphere: { scale: 0.25 },
            }
          );
          viewer.zoomTo();
          viewer.render();
        } finally {
          setIsRendering(false);
        }
      }, 50);
    },
    [sdfData]
  );

  useEffect(() => {
    if (viewerRef.current) loadModel(viewerRef.current);
  }, [loadModel]);

  // Handle ESC key
  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === "Escape") setIsOpen(false);
    };
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, []);

  return (
    <>
      {/* ------------------ COMPACT PREVIEW ------------------ */}
      <div className="group relative w-full h-80 rounded-xl border border-slate-800 bg-slate-950/50 overflow-hidden cursor-pointer hover:border-slate-700 transition-all">
        {isRendering && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-slate-950/50 backdrop-blur-sm">
            <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
          </div>
        )}

        <div ref={viewerRef} className="w-full h-full relative z-10" />

        {!isRendering && (
          <div
            onClick={() => setIsOpen(true)}
            className="absolute inset-0 z-30 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-slate-950/40 backdrop-blur-[2px]"
          >
            <button className="bg-slate-50 text-slate-900 px-4 py-2 rounded-full text-xs font-medium flex items-center gap-2 shadow-lg hover:bg-white transition-colors">
              <Maximize2 className="w-3 h-3" /> Fullscreen
            </button>
          </div>
        )}
      </div>

      {/* ------------------ FULLSCREEN MODAL ------------------ */}
      {isOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div
            className="relative w-[90vw] h-[85vh] max-w-6xl bg-slate-950 border border-slate-800 rounded-lg shadow-2xl flex flex-col animate-in zoom-in-95 duration-200"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 shrink-0">
              <h3 className="text-lg font-semibold text-slate-50">
                3D Structure Viewer
              </h3>
              <button
                onClick={() => setIsOpen(false)}
                className="rounded-sm opacity-70 ring-offset-slate-950 transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-slate-300 focus:ring-offset-2"
              >
                <X className="h-5 w-5 text-slate-50" />
                <span className="sr-only">Close</span>
              </button>
            </div>

            <div className="flex-1 p-4 overflow-hidden bg-slate-900/50 relative">
              <FullscreenViewer sdfData={sdfData} />
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function FullscreenViewer({ sdfData }) {
  const modalRef = useRef(null);

  useEffect(() => {
    if (!modalRef.current) return;
    modalRef.current.innerHTML = "";

    // FIX: Initialize without config
    const viewer = $3Dmol.createViewer(modalRef.current);

    // FIX: Set transparent background explicitly
    viewer.setBackgroundColor(0x000000, 1);

    viewer.addModel(sdfData, "sdf");
    viewer.setStyle({}, { stick: { radius: 0.25 }, sphere: { scale: 0.3 } });

    const t = setTimeout(() => {
      viewer.resize();
      viewer.zoomTo();
      viewer.render();
    }, 100);

    return () => clearTimeout(t);
  }, [sdfData]);

  return <div ref={modalRef} className="w-full h-full relative" />;
}
