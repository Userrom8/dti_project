import { useState } from "react";
import axios from "axios";
import {
  Loader2,
  Beaker,
  Dna,
  AlertCircle,
  X,
  CheckCircle2,
  ChevronDown,
  Info,
  BoxSelect,
  Atom,
  Cuboid,
} from "lucide-react";
import Molecule3DViewer from "./components/Mol3DViewer";
import Protein3DViewer from "./components/Protein3DViewer";

const API_URL = "http://127.0.0.1:8000/predict";

const EXAMPLES = [
  {
    name: "Ethanol + Ras-like",
    smiles: "CCO",
    protein: "MTEYKLVVVGAGGVGKSALTIQLIQNHFV...",
  },
  {
    name: "Aspirin + Example",
    smiles: "CC(=O)OC1=CC=CC=C1C(=O)O",
    protein: "MGSSHHHHHHSSGLVPRGSHMTEYKLVVVG...",
  },
];

/* --- UI Components (Simulating Shadcn) --- */

const Label = ({ children, htmlFor }) => (
  <label
    htmlFor={htmlFor}
    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 text-slate-200"
  >
    {children}
  </label>
);

const Input = (props) => (
  <input
    {...props}
    className={`flex h-10 w-full rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-50 ring-offset-slate-950 file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${
      props.className || ""
    }`}
  />
);

const Textarea = (props) => (
  <textarea
    {...props}
    className={`flex min-h-[80px] w-full rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-50 ring-offset-slate-950 placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${
      props.className || ""
    }`}
  />
);

const Button = ({ children, className, variant = "default", ...props }) => {
  const baseStyles =
    "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-slate-950 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50";

  const variants = {
    default: "bg-slate-50 text-slate-900 hover:bg-slate-50/90",
    destructive: "bg-red-900 text-slate-50 hover:bg-red-900/90",
    outline:
      "border border-slate-800 bg-slate-950 hover:bg-slate-800 hover:text-slate-50",
    ghost: "hover:bg-slate-800 hover:text-slate-50",
    link: "text-slate-50 underline-offset-4 hover:underline",
  };

  return (
    <button
      className={`${baseStyles} ${variants[variant]} ${className || ""}`}
      {...props}
    >
      {children}
    </button>
  );
};

const Card = ({ children, className }) => (
  <div
    className={`rounded-xl border border-slate-800 bg-slate-950 text-slate-50 shadow-sm ${
      className || ""
    }`}
  >
    {children}
  </div>
);

/* --- Custom Tooltip Component --- */
const Tooltip = ({ content, children }) => (
  <div className="relative inline-flex items-center group">
    {children}
    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block w-max max-w-xs px-2 py-1 bg-slate-800 border border-slate-700 text-xs text-slate-200 rounded shadow-xl z-50 animate-in fade-in zoom-in-95 duration-200 pointer-events-none">
      {content}
      <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-[1px] border-4 border-transparent border-t-slate-700" />
    </div>
  </div>
);

/* --- Skeleton Loader --- */
function ResultSkeleton() {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950 p-6 space-y-4 animate-pulse">
      <div className="space-y-2">
        <div className="h-4 bg-slate-800 rounded w-1/3" />
        <div className="h-4 bg-slate-800 rounded w-2/3" />
      </div>
      <div className="h-12 bg-slate-800 rounded w-full" />
      <div className="h-64 bg-slate-800 rounded-xl w-full" />
    </div>
  );
}

export default function App() {
  // New state for toggling input mode
  const [inputType, setInputType] = useState("smiles"); // "smiles" or "name"

  const [smiles, setSmiles] = useState("");
  const [drugName, setDrugName] = useState(""); // New state for drug name
  const [protein, setProtein] = useState("");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // 3D Viewer States
  const [sdf3d, setSdf3d] = useState(null);
  const [pdb3d, setPdb3d] = useState(null);
  const [sdfError, setSdfError] = useState(false);

  // fetchDrug3D can handle both SMILES and Names via the Cactus API
  const fetchDrug3D = async (inputVal) => {
    setSdfError(false);
    try {
      const url = `https://cactus.nci.nih.gov/chemical/structure/${encodeURIComponent(
        inputVal
      )}/file?format=sdf&get3d=true`;

      const resp = await fetch(url);
      if (!resp.ok) throw new Error("3D structure unavailable");

      const text = await resp.text();
      // Basic validation to check if returned text looks like SDF
      if (!text.includes("V2000") && !text.includes("V3000")) {
        throw new Error("Invalid SDF format");
      }
      setSdf3d(text);
    } catch (err) {
      console.error("3D fetch error:", err);
      setSdf3d(null);
      setSdfError(true);
    }
  };

  // 2. Fetch Protein 3D (ESMFold API)
  const fetchProtein3D = async (rawSequence) => {
    try {
      // CLEAN THE SEQUENCE: Remove FASTA headers (lines starting with >) and all whitespace
      const cleanSequence = rawSequence
        .split("\n")
        .filter((line) => !line.trim().startsWith(">")) // Remove header lines
        .join("")
        .replace(/\s+/g, "") // Remove spaces/newlines
        .toUpperCase();

      if (cleanSequence.length > 400) {
        console.warn("Sequence too long for instant 3D preview");
        setPdb3d(null);
        return;
      }

      const url = "https://api.esmatlas.com/foldSequence/v1/pdb/";
      const resp = await fetch(url, {
        method: "POST",
        body: cleanSequence,
      });

      if (!resp.ok) throw new Error("Protein folding failed");
      const text = await resp.text();

      // Basic validation: PDB files should start with HEADER or contain ATOM records
      if (!text.includes("ATOM") && !text.includes("HEADER")) {
        throw new Error("Invalid PDB data");
      }

      setPdb3d(text);
    } catch (err) {
      console.error("Protein 3D fetch error:", err);
      setPdb3d(null);
    }
  };

  const handlePickExample = (e) => {
    const name = e.target.value;
    if (!name) return;
    const ex = EXAMPLES.find((x) => x.name === name);
    if (ex) {
      setInputType("smiles"); // Default to SMILES for examples
      setSmiles(ex.smiles);
      setDrugName("");
      setProtein(ex.protein);
      setResult(null);
      setError("");
      setSdf3d(null);
      setSdfError(false);
    }
  };

  const handleClear = () => {
    setSmiles("");
    setDrugName("");
    setProtein("");
    setResult(null);
    setError("");
    setSdf3d(null);
    setSdfError(false);
  };

  const handlePredict = async () => {
    setError("");
    setResult(null);
    setSdf3d(null);
    setSdfError(false);

    // Determine the active input value based on the selected mode
    const activeInput = inputType === "smiles" ? smiles : drugName;

    if (!activeInput.trim() || !protein.trim()) {
      setError(
        `Protein Sequence and ${
          inputType === "smiles" ? "SMILES" : "Drug Name"
        } are required.`
      );
      return;
    }

    try {
      setLoading(true);

      // --- Parallel Fetching ---
      // 1. Fetch Drug Structure
      const drugPromise = fetchDrug3D(activeInput);
      // 2. Fetch Protein Structure (ESMFold)
      const proteinPromise = fetchProtein3D(protein);
      // 3. Fetch Prediction (Your Backend)
      const payload = {
        protein,
        ...(inputType === "smiles" ? { smiles } : { drug_name: drugName }),
      };
      const predictionPromise = axios.post(API_URL, payload, {
        timeout: 120000,
      });

      // Wait for everything
      const [_, __, resp] = await Promise.allSettled([
        drugPromise,
        proteinPromise,
        predictionPromise,
      ]);

      // Check prediction result (the most important part)
      if (resp.status === "fulfilled") {
        if (resp.value.data && resp.value.data.affinity !== undefined) {
          setResult(Number(resp.value.data.affinity));
        } else if (resp.value.data && resp.value.data.error) {
          setError(resp.value.data.error);
        } else {
          setError("Unexpected API response.");
        }
      } else {
        throw new Error(
          resp.reason?.response?.data?.error ||
            resp.reason?.message ||
            "Prediction failed."
        );
      }
    } catch (err) {
      setError(err.message || "An error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 py-12 px-4 font-sans selection:bg-slate-300 selection:text-slate-900">
      <div className="max-w-2xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center p-3 bg-slate-900 rounded-full mb-2 border border-slate-800">
            <Dna className="w-8 h-8 text-slate-200" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-100">
            Drug–Target Interaction
          </h1>
          <p className="text-slate-400 max-w-md mx-auto text-sm">
            Predict binding affinity using GNNs for drugs and Transformers for
            proteins.
          </p>
        </div>

        {/* Input Card */}
        <Card>
          <div className="flex flex-col space-y-1.5 p-6 pb-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold leading-none tracking-tight">
                Analysis Parameters
              </h3>
              {(smiles || drugName || protein) && (
                <Button
                  variant="ghost"
                  className="h-8 px-2 text-slate-400 hover:text-red-400"
                  onClick={handleClear}
                >
                  <X className="w-4 h-4 mr-1" /> Clear
                </Button>
              )}
            </div>
          </div>

          <div className="p-6 pt-0 space-y-4">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label>Load Example</Label>
                <Tooltip content="Pre-filled examples to test the model">
                  <Info className="w-3.5 h-3.5 text-slate-500 cursor-help" />
                </Tooltip>
              </div>
              <div className="relative">
                <select
                  onChange={handlePickExample}
                  className="flex h-10 w-full items-center justify-between rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-300 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 appearance-none text-slate-50"
                  defaultValue=""
                >
                  <option value="" disabled>
                    Select a preset...
                  </option>
                  {EXAMPLES.map((ex) => (
                    <option key={ex.name} value={ex.name}>
                      {ex.name}
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-3 h-4 w-4 opacity-50 pointer-events-none" />
              </div>
            </div>

            <div className="grid gap-4 pt-2">
              {/* Input Mode Toggle */}
              <div className="space-y-2">
                <Label>Input Mode</Label>
                <div className="flex gap-6">
                  <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-300 hover:text-white transition-colors">
                    <input
                      type="radio"
                      name="inputType"
                      checked={inputType === "smiles"}
                      onChange={() => setInputType("smiles")}
                      className="accent-blue-500"
                    />
                    SMILES String
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-300 hover:text-white transition-colors">
                    <input
                      type="radio"
                      name="inputType"
                      checked={inputType === "name"}
                      onChange={() => setInputType("name")}
                      className="accent-blue-500"
                    />
                    Drug Name
                  </label>
                </div>
              </div>

              {/* Conditional Input Rendering */}
              {inputType === "smiles" ? (
                <div className="space-y-2">
                  <Label htmlFor="smiles">
                    <div className="flex items-center gap-2">
                      <Tooltip content="Simplified Molecular Input Line Entry System">
                        <Beaker className="w-3.5 h-3.5 text-blue-400" />
                      </Tooltip>
                      Drug SMILES
                    </div>
                  </Label>
                  <Input
                    id="smiles"
                    placeholder="e.g. CCO"
                    value={smiles}
                    onChange={(e) => setSmiles(e.target.value)}
                    className="font-mono text-sm"
                  />
                </div>
              ) : (
                <div className="space-y-2">
                  <Label htmlFor="drugName">
                    <div className="flex items-center gap-2">
                      <Tooltip content="Common or Generic Drug Name">
                        <Beaker className="w-3.5 h-3.5 text-purple-400" />
                      </Tooltip>
                      Drug Name
                    </div>
                  </Label>
                  <Input
                    id="drugName"
                    placeholder="e.g. Aspirin"
                    value={drugName}
                    onChange={(e) => setDrugName(e.target.value)}
                    className="text-sm"
                  />
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="protein">
                  <div className="flex items-center gap-2">
                    <Tooltip content="Amino acid sequence of the target protein">
                      <Dna className="w-3.5 h-3.5 text-green-400" />
                    </Tooltip>
                    Protein Sequence
                  </div>
                </Label>
                <Textarea
                  id="protein"
                  rows={5}
                  placeholder="Paste FASTA or raw sequence..."
                  value={protein}
                  onChange={(e) => setProtein(e.target.value)}
                  className="font-mono text-xs leading-relaxed resize-none"
                />
                <div className="text-xs text-right text-slate-500">
                  Length: {protein.length} aa
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center p-6 pt-0">
            <Button
              className="w-full text-md font-semibold h-11"
              onClick={handlePredict}
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Analyzing...
                </>
              ) : (
                "Predict Affinity"
              )}
            </Button>
          </div>
        </Card>

        {/* Error Alert */}
        {error && (
          <div className="relative w-full rounded-lg border border-red-900/50 p-4 [&>svg~*]:pl-7 bg-red-900/10 text-red-500">
            <AlertCircle className="absolute left-4 top-4 h-4 w-4 text-red-500" />
            <h5 className="mb-1 font-medium leading-none tracking-tight">
              Error
            </h5>
            <div className="text-sm opacity-90">{error}</div>
          </div>
        )}

        {/* Loading Skeleton */}
        {loading && <ResultSkeleton />}

        {/* Result Card */}
        {!loading && result !== null && (
          <Card className="overflow-hidden border-slate-800 shadow-lg transition-all animate-in fade-in slide-in-from-bottom-4">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-green-500" />

            <div className="flex flex-col space-y-1.5 p-6">
              <h3 className="font-semibold leading-none tracking-tight flex items-center gap-2 text-slate-50">
                <CheckCircle2 className="w-5 h-5 text-green-500" />
                Prediction Result
              </h3>
              <p className="text-sm text-slate-400">
                Calculated binding affinity (higher indicates stronger binding)
              </p>
            </div>

            <div className="p-6 pt-0 space-y-6">
              <div className="flex items-baseline justify-between bg-slate-900/50 p-4 rounded-lg border border-slate-800/50">
                <span className="text-sm font-medium text-slate-400">
                  Affinity Score
                </span>
                <span className="text-4xl font-extrabold text-slate-50 tracking-tight">
                  {result.toFixed(4)}
                </span>
              </div>

              {/* 3D Viewer Section */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-sm font-medium text-slate-300">
                  <Atom className="w-4 h-4 text-blue-400" /> Molecular Structure
                </div>

                {sdf3d ? (
                  <>
                    <Molecule3DViewer sdfData={sdf3d} />
                    <p className="text-[10px] uppercase tracking-wider text-center text-slate-500">
                      Interactive Viewer • Click to expand
                    </p>
                  </>
                ) : sdfError ? (
                  /* Error Placeholder for 3D Viewer */
                  <div className="w-full h-80 rounded-xl border border-dashed border-slate-800 bg-slate-950/30 flex flex-col items-center justify-center text-slate-500 gap-3">
                    <BoxSelect className="w-10 h-10 opacity-50" />
                    <div className="text-center px-6">
                      <p className="text-sm font-medium text-slate-400">
                        3D Structure Unavailable
                      </p>
                      <p className="text-xs mt-1">
                        Could not generate a 3D model for this{" "}
                        {inputType === "smiles" ? "SMILES string" : "Drug Name"}
                        .
                      </p>
                    </div>
                  </div>
                ) : null}
              </div>

              {/* 2. Protein Viewer */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-sm font-medium text-slate-300">
                  <Cuboid className="w-4 h-4 text-green-400" /> Target Structure
                </div>
                {pdb3d ? (
                  <Protein3DViewer pdbData={pdb3d} />
                ) : (
                  <div className="w-full h-80 rounded-xl border border-dashed border-slate-800 bg-slate-950/30 flex flex-col items-center justify-center text-slate-500 gap-3">
                    <Dna className="w-10 h-10 opacity-50" />
                    <div className="text-center px-6">
                      <p className="text-sm font-medium text-slate-400">
                        Structure Unavailable
                      </p>
                      <p className="text-[10px] mt-1 text-slate-600">
                        Sequence may be too long for instant preview.
                      </p>
                    </div>
                  </div>
                )}
              </div>

              <div className="text-xs text-slate-500 border-t border-slate-800 pt-4 grid grid-cols-2 gap-4">
                <div>
                  <span className="block font-semibold text-slate-300 mb-1">
                    Drug Input
                  </span>
                  <p className="truncate font-mono opacity-80">
                    {inputType === "smiles" ? smiles : drugName}
                  </p>
                </div>
                <div className="text-right">
                  <span className="block font-semibold text-slate-300 mb-1">
                    Target Input
                  </span>
                  <p className="font-mono opacity-80">
                    {protein.length} amino acids
                  </p>
                </div>
              </div>
            </div>
          </Card>
        )}

        <footer className="text-center text-xs text-slate-600 py-4">
          Local API Endpoint:{" "}
          <code className="bg-slate-900 px-1 py-0.5 rounded border border-slate-800 text-slate-400">
            /predict
          </code>
        </footer>
      </div>
    </div>
  );
}
