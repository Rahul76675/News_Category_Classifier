import React, { useState } from 'react';
import { Send } from 'lucide-react';

function App() {
  const [newsText, setNewsText] = useState('');
  const [prediction, setPrediction] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const categories = [
    'POLITICS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY', 'SPORTS', 
    'PARENTING', 'HEALTHY LIVING', 'QUEER VOICES', 'FOOD & DRINK', 'BUSINESS',
    'COMEDY', 'TECH', 'ENVIRONMENT', 'EDUCATION', 'CRIME', 'MEDIA',
    'WEIRD NEWS', 'GREEN', 'IMPACT', 'WORLDPOST', 'RELIGION', 'SCIENCE',
    'CULTURE & ARTS', 'COLLEGE', 'LATINO VOICES', 'WEDDINGS', 'BLACK VOICES',
    'WOMEN', 'HOME & LIVING', 'PARENTS', 'DIVORCE', 'WORLD NEWS', 'U.S. NEWS'
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      // This would be replaced with actual API call to your model endpoint
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: newsText }),
      });

      const data = await response.json();
      setPrediction(data.category);
    } catch (error) {
      console.error('Error predicting category:', error);
      setPrediction('Error occurred while predicting');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="bg-white rounded-lg shadow-xl p-6 space-y-6">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-gray-900">News Category Classifier</h1>
            <p className="mt-2 text-gray-600">Enter a news article headline and description to predict its category</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="newsText" className="block text-sm font-medium text-gray-700">
                News Text
              </label>
              <textarea
                id="newsText"
                rows={6}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                placeholder="Enter news headline and description here..."
                value={newsText}
                onChange={(e) => setNewsText(e.target.value)}
              />
            </div>

            <button
              type="submit"
              disabled={isLoading || !newsText.trim()}
              className="w-full flex justify-center items-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                'Predicting...'
              ) : (
                <>
                  Predict Category
                  <Send className="ml-2 h-4 w-4" />
                </>
              )}
            </button>
          </form>

          {prediction && (
            <div className="mt-6 p-4 bg-gray-50 rounded-md">
              <h2 className="text-lg font-medium text-gray-900">Predicted Category:</h2>
              <p className="mt-2 text-xl font-semibold text-indigo-600">{prediction}</p>
            </div>
          )}

          <div className="mt-8">
            <h3 className="text-lg font-medium text-gray-900 mb-3">Available Categories:</h3>
            <div className="flex flex-wrap gap-2">
              {categories.map((category) => (
                <span
                  key={category}
                  className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800"
                >
                  {category}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;