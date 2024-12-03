import * as tf from '@tensorflow/tfjs-node';
import { LOOKUP, PAD, START, UNKNOWN } from '../../../../public/dictionary'; // Adjust the path to your dictionary.js
import path from 'path';

// Preprocess function
function preprocessText(text) {
    const lowerText = text.toLowerCase();
    const tokens = lowerText.split(/\s+/); // Split by spaces
    const tokenIndices = tokens.map((token) => LOOKUP[token] || UNKNOWN); // Use LOOKUP or UNKNOWN for unknown words

    // Pad or truncate the sequence to length 20
    const maxLen = 20;
    if (tokenIndices.length > maxLen) {
        tokenIndices.splice(maxLen); // Truncate
    } else if (tokenIndices.length < maxLen) {
        while (tokenIndices.length < maxLen) {
            tokenIndices.push(PAD); // Pad
        }
    }

    return tf.tensor2d([tokenIndices], [1, maxLen]);
}

// Load model (singleton to avoid reloading on every request)
let model = null;

async function getModel() {
  if (!model) {
      const modelUrl = process.env.NEXT_PUBLIC_MODEL_URL; // Access environment variable
      model = await tf.loadLayersModel(modelUrl); // Use the URL from .env
      console.log('Model loaded successfully!');
  }
  return model;
}
export async function POST(req) {
    try {
        // Ensure that the request method is POST
        if (req.method !== 'POST') {
            return new Response(JSON.stringify({ error: 'Method not allowed' }), {
                status: 405,
                headers: { 'Content-Type': 'application/json' }
            });
        }

        const { text } = await req.json();
        if (!text) {
            return new Response(JSON.stringify({ error: 'Text is required' }), {
                status: 400,
                headers: { 'Content-Type': 'application/json' }
            });
        }

        // Load the model
        const model = await getModel();

        // Preprocess input text
        const inputTensor = preprocessText(text);

        // Make prediction
        const prediction = model.predict(inputTensor);
        const spamProbability = prediction.dataSync()[0];

        // Send response
        return new Response(
            JSON.stringify({
                spamProbability,
                isSpam: spamProbability > 0.8,
            }),
            {
                status: 200,
                headers: { 'Content-Type': 'application/json' }
            }
        );
    } catch (error) {
        console.error('Error occurred:', error);
        return new Response(JSON.stringify({ error: 'Internal server error' }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}
