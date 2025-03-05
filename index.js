// index.js (Minimal "Hello World" Worker in plain JavaScript)

function handleRequest(request) { // Use a simple function, not async for basic test
    return new Response('Hello World from Minimal Worker!', { 
        status: 200,
        headers: { 'content-type': 'text/plain' }
    });
}

addEventListener('fetch', event => { // Use addEventListener for fetch (alternative to export default fetch in some cases)
    event.respondWith(handleRequest(event.request));
});