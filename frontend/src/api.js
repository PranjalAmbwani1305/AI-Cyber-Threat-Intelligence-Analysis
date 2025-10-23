const BASE_URL = "https://YOUR_BACKEND_URL"; 

export async function analyzeNER(text) {
  const res = await fetch(`${BASE_URL}/ner`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return res.json();
}

export async function analyzeSentiment(text) {
  const res = await fetch(`${BASE_URL}/sentiment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return res.json();
}

export async function analyzeCluster(texts) {
  const res = await fetch(`${BASE_URL}/cluster`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts }),
  });
  return res.json();
}
