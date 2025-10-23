import { useState } from "react";

function App() {
  const [text, setText] = useState("");
  const [ner, setNer] = useState("");
  const [sentiment, setSentiment] = useState("");
  const [cluster, setCluster] = useState("");

  const callAPI = async (endpoint, setter) => {
    const res = await fetch(`https://your-backend.vercel.app/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    setter(JSON.stringify(data, null, 2));
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>AI Cyber Threat Analysis</h1>
      <textarea
        rows="5"
        cols="50"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div>
        <button onClick={() => callAPI("ner", setNer)}>Run NER</button>
        <button onClick={() => callAPI("sentiment", setSentiment)}>Sentiment</button>
        <button onClick={() => callAPI("cluster", setCluster)}>Clustering</button>
      </div>
      <div>
        <h3>NER Result:</h3>
        <pre>{ner}</pre>
        <h3>Sentiment Result:</h3>
        <pre>{sentiment}</pre>
        <h3>Cluster Result:</h3>
        <pre>{cluster}</pre>
      </div>
    </div>
  );
}

export default App;
