export default {
    async fetch(request, env) {
      const url = new URL(request.url);
  
      // Handle CORS preflight requests
      if (request.method === 'OPTIONS') {
        return new Response(null, {
          headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
          },
        });
      }
  
      // Handle API requests
      if (request.method === 'POST' && url.pathname === '/') { // Corrected to only handle POST on root path
        try {
          const sessionId = url.searchParams.get('sessionId') || 'default';
          const data = await request.json();
          const input = data.input;
  
          if (!input) {
            return new Response(JSON.stringify({ error: 'Input is required' }), {
              status: 400,
              headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
              }
            });
          }
  
          // Choose which model to use based on query parameter
          const modelType = url.searchParams.get('model') || 'llama';
  
          let llamaResponse;
          if (modelType === 'llama') {
            llamaResponse = await fetch(
              `https://api.cloudflare.com/client/v4/accounts/a48f8548d25dd84eccf9c90840e539aa/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast`,
              {
                headers: {
                  "Authorization": "Bearer BYokKCK9_t_R2KPZmhvsdCpyHgYd8iNUJgQNNkoU",
                  "Content-Type": "application/json"
                },
                method: "POST",
                body: JSON.stringify({
                  messages: [
                    { role: "system", content: "You are Hextrix AI, a helpful and knowledgeable assistant." },
                    { role: "user", content: input }
                  ],
                }),
              }
            );
          } else { // Assuming default to llama if model is not 'llama'
            llamaResponse = await fetch(
              `https://api.cloudflare.com/client/v4/accounts/a48f8548d25dd84eccf9c90840e539aa/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast`, // Defaulting to llama
              {
                headers: {
                  "Authorization": "Bearer BYokKCK9_t_R2KPZmhvsdCpyHgYd8iNUJgQNNkoU",
                  "Content-Type": "application/json"
                },
                method: "POST",
                body: JSON.stringify({
                  messages: [
                    { role: "system", content: "You are Hextrix AI, a helpful and knowledgeable assistant." },
                    { role: "user", content: input }
                  ],
                }),
              }
            );
          }
  
  
          if (!llamaResponse.ok) {
            throw new Error(`Llama API error! status: ${llamaResponse.status}`);
          }
  
          const result = await llamaResponse.json();
  
          if (!result.success) {
            throw new Error(result.errors?.[0]?.message || 'Llama API error');
          }
  
          return new Response(JSON.stringify({
            response: result.result.response,
            history: []
          }), {
            headers: {
              'Content-Type': 'application/json',
              'Access-Control-Allow-Origin': '*'
            }
          });
        } catch (error) {
          return new Response(JSON.stringify({ error: error.message }), {
            status: 500,
            headers: {
              'Content-Type': 'application/json',
              'Access-Control-Allow-Origin': '*'
            }
          });
        }
      }
  
      // Serve static assets
      if (url.pathname.startsWith('/assets/')) {
        return env.ASSETS.fetch(request);
      }
  
      // Serve HTML pages - Corrected path matching for index and memory
      if (url.pathname === '/' || url.pathname === '/index.html') {
        return env.ASSETS.fetch(new Request('/index.html', request)); // Simplified path
      }
  
      if (url.pathname === '/memory.html' || url.pathname === '/enhanced-neural-memory-map.html') { // Added enhanced-neural-memory-map.html
        return env.ASSETS.fetch(new Request(url.pathname, request)); // Use original pathname for memory and enhanced memory pages
      }
  
      // Default: return 405 for unsupported methods
      return new Response('Method not allowed', {
        status: 405,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'text/plain'
        }
      });
    }
  };