import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [smiles, setSmiles] = useState("");
  const [protein, setProtein] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ⚠ IMPORTANT: Replace with your Render backend API URL
  // const API_URL = "https://your-backend.onrender.com/predict";
  const API_URL = "http://127.0.0.1:8000/predict";

  const handlePredict = async () => {
    setError("");
    setResult(null);

    if (!smiles || !protein) {
      setError("Both SMILES and Protein Sequence are required!");
      return;
    }

    try {
      setLoading(true);
      const response = await axios.post(API_URL, {
        smiles,
        protein,
      });

      if (response.data.affinity !== undefined) {
        setResult(response.data.affinity.toFixed(4));
      } else {
        setError("Unexpected API response");
      }
    } catch (err) {
      setError("Prediction failed. Check API URL or input format.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Drug–Target Interaction Predictor</h1>
      <p className="subtitle">Powered by GNN + ESM Protein Transformer</p>

      <div className="input-section">
        <label>Drug SMILES</label>
        <input
          type="text"
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          placeholder="e.g. CCO"
        />

        <label>Protein Sequence (FASTA or plain)</label>
        <textarea
          rows="6"
          value={protein}
          onChange={(e) => setProtein(e.target.value)}
          placeholder="Paste protein sequence here..."
        />
      </div>

      <button onClick={handlePredict} disabled={loading}>
        {loading ? "Predicting..." : "Predict Affinity"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result-box">
          <h2>Predicted Binding Affinity</h2>
          <p className="affinity-value">{result}</p>
        </div>
      )}
    </div>
  );
}

export default App;
