import { useState } from "react";
import axios from "axios";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAsk = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await axios.post(
        import.meta.env.VITE_API_URL + "/qa/ask",
        {
          question: question,
          top_k: 3
        }
      );
      setAnswer(response.data.answer);
    } catch (err) {
      setError("Failed to get answer. Check backend or network.");
      console.error(err);
    }
    setLoading(false);
  };

  return (
    <div className="max-w-xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">AI Support Assistant</h1>
      <textarea
        rows="5"
        className="w-full p-2 border rounded"
        placeholder="Type your question..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      ></textarea>
      <button
        className="mt-3 bg-blue-500 text-white px-4 py-2 rounded"
        onClick={handleAsk}
        disabled={loading}
      >
        {loading ? "Loading..." : "Ask"}
      </button>
      <h1 className="text-3xl font-bold text-blue-600">Tailwind is Working!</h1>


      {error && <p className="text-red-500 mt-3">{error}</p>}

      {answer && (
        <div className="mt-5 p-4 border rounded bg-gray-100">
          <h2 className="font-semibold">Answer:</h2>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}

export default App;
