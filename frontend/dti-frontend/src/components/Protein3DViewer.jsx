import { useEffect, useRef, useState, useCallback } from "react";
import * as $3Dmol from "3dmol";
import { Maximize2, X, Loader2 } from "lucide-react";

export default function Protein3DViewer({ pdbData }) {
  const viewerRef = useRef(null);
  const [isOpen, setIsOpen] = useState(false);
  const [isRendering, setIsRendering] = useState(true);

  const loadModel = useCallback(
    (target) => {
      if (!pdbData || !target) return;

      setIsRendering(true);

      setTimeout(() => {
        try {
          // 1. Clear previous content
          target.innerHTML = "";

          // 2. Initialize Viewer (No config object to avoid warnings)
          const viewer = $3Dmol.createViewer(target);

          // 3. Set Transparent Background
          viewer.setBackgroundColor(0x000000, 1);

          // 4. Add Model
          viewer.addModel(pdbData, "pdb");

          // 5. Apply Style
          // We use a 'cartoon' style with specific settings.
          // If the PDB lacks secondary structure headers, standard cartoon might fail.
          // We add 'tube' as a robust fallback that looks like a cartoon backbone.
          viewer.setStyle(
            {},
            {
              cartoon: {
                color: "spectrum",
                style: "trace", // 'trace' or 'oval' creates a smooth tube even without headers
                thickness: 1.0,
              },
            }
          );

          // 6. Initial Render & Zoom
          viewer.zoomTo();
          viewer.render();

          // 7. Force Re-zoom (Fixes "empty view" bug on resize)
          setTimeout(() => {
            viewer.resize();
            viewer.zoomTo();
            viewer.render();
          }, 200);
        } catch (e) {
          console.error("Viewer Render Error:", e);
        } finally {
          setIsRendering(false);
        }
      }, 50);
    },
    [pdbData]
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
            <Loader2 className="w-8 h-8 text-green-500 animate-spin" />
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
                Protein Structure (Predicted)
              </h3>
              <button
                onClick={() => setIsOpen(false)}
                className="rounded-sm opacity-70 transition-opacity hover:opacity-100 focus:outline-none"
              >
                <X className="h-5 w-5 text-slate-50" />
              </button>
            </div>

            <div className="flex-1 p-4 overflow-hidden bg-slate-900/50 relative">
              <FullscreenProteinViewer pdbData={pdbData} />
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function FullscreenProteinViewer({ pdbData }) {
  const modalRef = useRef(null);

  useEffect(() => {
    if (!modalRef.current) return;

    // 1. Clear & Init
    modalRef.current.innerHTML = "";
    const viewer = $3Dmol.createViewer(modalRef.current);
    viewer.setBackgroundColor(0x000000, 1);

    // 2. Add Model & Style (Trace Cartoon)
    viewer.addModel(pdbData, "pdb");
    viewer.setStyle(
      {},
      {
        cartoon: {
          color: "spectrum",
          style: "trace",
          thickness: 1.0,
        },
      }
    );

    // 3. Render Loop (Aggressive)
    const t1 = setTimeout(() => {
      viewer.resize();
      viewer.zoomTo();
      viewer.render();
    }, 50);

    const t2 = setTimeout(() => {
      viewer.zoomTo();
      viewer.render();
    }, 300);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [pdbData]);

  return <div ref={modalRef} className="w-full h-full relative" />;
}
