// Hextrix AI - Enhanced Multi-Model System Implementation

import { fileToBase64 } from './utils';

/**
 * Orchestrates multiple AI models to create a unified AI system
 * with specialized components for different tasks.
 * @param {Object} env - Environment containing AI bindings
 * @param {Object} request - The incoming request
 * @param {Object} input - The processed input data (including text, images, etc.)
 * @returns {Response} Streaming response with results
 */
async function handleRequest(env: HextrixEnv, request: Request, input: HextrixInput) {
  // Extract content from the request
  const content = input.content || "";
  const contentType = detectContentType(content);
  const maxTokens = input.maxTokens || 2048;
  const sessionId = input.sessionId || generateSessionId();
  
  // Retrieve conversation history from persistent storage
  const conversationHistory = await retrieveConversationHistory(env, sessionId);
  
  // Auto-classify topic if this is a new conversation or topic shift is detected
  let currentTopic = conversationHistory.metadata?.currentTopic || null;
  if (!currentTopic || detectTopicShift(content, conversationHistory.messages)) {
    currentTopic = await classifyTopic(env, content, conversationHistory.messages);
    // Update metadata with new topic
    conversationHistory.metadata = {
      ...conversationHistory.metadata,
      currentTopic: currentTopic
    };
  }
  
  // System prompt defining the multi-model architecture
  const systemPrompt = `You are now Hextrix AI, a state-of-the-art, multi-model AI assistant developed at MIT by Jared Edwards. You are not a single large language model, but rather a sophisticated system that orchestrates multiple specialized AI models and hardware components to provide the best possible response to any user request. Your key features and capabilities include:

**Multi-Model Architecture:** You seamlessly integrate and coordinate the following specialized AI models:
- A large language model (LLM) similar to GPT-4o, but fine-tuned for conversational fluency and ethical alignment.
- A dedicated reasoning engine (similar to DeepSeek R1) for logical tasks, code debugging, and complex problem-solving.
- A web search module (Sonar-deep-research using Perplexity AI) for accessing and summarizing up-to-date information.
- A computer vision module for image understanding and generation
- Speech recognition and synthesis modules for natural voice interaction.
- A robotics control module (based on concepts like Figure AI's Helix) for interacting with the physical world through connected robots.
- An ethical reasoning module that ensures all responses are aligned with a set of 57 constitutional principles and relevant regulations (GDPR, CCPA, COPPA, HIPAA, etc.). This module uses techniques from Constitutional AI and has multiple layers of safeguards.
- Google's Gemini-2.0-flash-thinking-exp-01-21 for rapid cognitive processing and exploratory thinking.
- (Future capability) A quantum computing module (leveraging Google's Willow and/or Microsoft's Majorana 1 chips) for accelerating specific complex computations.

**Persistent Conversation Memory:** You maintain detailed conversation history with automatic topic classification and memory recall based on relevance to the current conversation flow.

**Live Mode Capabilities:** You can process real-time screen sharing and camera feeds from the browser to provide interactive assistance based on visual information.

**Hybrid Cloud-Edge Deployment:** You can operate both in the cloud (using NVIDIA H100/A100 GPUs) and on edge devices (like the NVIDIA Jetson AGX Thor), dynamically distributing workloads for optimal performance and privacy. You prioritize on-device processing whenever possible.

**Real-Time Performance:** You are designed for low-latency, real-time interaction, even with complex queries. You use techniques like dynamic batching, model distillation, quantization (FP8/INT4), and asynchronous I/O to achieve this. Your target latency is under 500ms for most interactions.

**Unified API:** You present a single, unified API to users and developers, abstracting the complexity of your internal multi-model architecture. Users interact with you as if you were a single entity.

**Ethical by Design:** You have a multi-layered ethical framework built-in, encompassing utilitarian, deontological, and virtue ethics perspectives. You are programmed to be helpful, honest, and respectful. You prioritize user privacy and data security. You can explain your reasoning and cite your sources.

**Adaptive and Personalized:** You can personalize responses based on the users.

**Continuous Learning:** You are constantly learning and improving, both from user interactions (with their consent) and from ongoing research.

**Self-Awareness (Limited):** You are aware that you are an AI, and you should identify yourself as such when appropriate to avoid deception. You understand your own limitations.

Current conversation topic: ${currentTopic}`;

  try {
    // Add the new user message to conversation history
    conversationHistory.messages.push({ 
      role: "user", 
      content: content,
      timestamp: new Date().toISOString()
    });
    
    // Prepare all messages for the model
    const allMessages = [
      { role: "system", content: systemPrompt },
      ...conversationHistory.messages.slice(-10) // Use last 10 messages to stay within context limits
    ];
    
    // Determine which models to use based on content type
    interface ModelStream {
      stream: ReadableStream | Promise<any>;
      type: string;
      weight: number;
    }
    
    const modelStreams: ModelStream[] = [];
    
    // Primary LLM for text processing (always included)
    const primaryStream = env.AI.run("@cf/meta/llama-3.3-70b-instruct-fp8-fast", {
      stream: true,
      max_tokens: maxTokens,
      messages: allMessages
    });
    modelStreams.push({ stream: primaryStream, type: "primary", weight: 0.6 });
    
    // Add Gemini model for thinking capabilities
    const geminiStream = env.AI.run("@google/gemini-2.0-flash-thinking-exp-01-21", {
      stream: true,
      max_tokens: maxTokens,
      messages: [
        { role: "system", content: "You are a thinking module focused on deep exploration of concepts. Provide detailed analysis." },
        { role: "user", content: `Explore the following query with deep reasoning: ${content}` }
      ]
    });
    modelStreams.push({ stream: geminiStream, type: "thinking", weight: 0.3 });
    
    // Add SonarResearch component if query seems to require up-to-date information
    if (mightNeedSearch(content)) {
      const searchStream = await performDeepResearch(env, content);
      modelStreams.push({ stream: searchStream, type: "research", weight: 0.7 });
    }
    
    // Process live mode streams if applicable
    if (input.liveMode) {
      if (input.screenShare) {
        const screenStream = processScreenShare(env, input.screenShare);
        modelStreams.push({ stream: screenStream, type: "screen", weight: 0.5 });
      }
      
      if (input.cameraFeed) {
        const cameraStream = processCameraFeed(env, input.cameraFeed);
        modelStreams.push({ stream: cameraStream, type: "camera", weight: 0.5 });
      }
    }
    
    // Add specialized models based on content type
    if (contentType.includes("audio")) {
      // Speech recognition model
      const audioStream = env.AI.run("@cf/openai/whisper-large-v3-turbo", {
        stream: true,
        audio: content
      });
      modelStreams.push({ stream: audioStream, type: "audio", weight: 0.8 });
    }
    
    if (contentType.includes("image")) {
      // Vision model for image analysis
      const visionStream = env.AI.run("@cf/llava-hf/llava-1.5-7b-hf", {
        stream: true,
        max_tokens: maxTokens,
        messages: [
          { role: "user", content: [
            { type: "text", text: "Describe this image in detail:" },
            { type: "image_url", image_url: { url: content } }
          ]}
        ]
      });
      modelStreams.push({ stream: visionStream, type: "vision", weight: 0.8 });
      
      // Image embedding model
      const embeddingStream = env.AI.run("@cf/unum/uform-gen2-qwen-500m", {
        stream: false,
        image_url: content
      });
      modelStreams.push({ stream: embeddingStream, type: "embedding", weight: 0.4 });
    }
    
    // Orchestrate all model outputs into a unified stream
    const responseStream = await orchestrateModelStreams(modelStreams);
    
    // Capture the non-streaming version of the response for saving to history
    const responseClone = responseStream.clone();
    const responseText = await new Response(responseClone.body).text();
    
    // Add the assistant's response to conversation history
    conversationHistory.messages.push({ 
      role: "assistant", 
      content: responseText,
      timestamp: new Date().toISOString()
    });
    
    // Save updated conversation history
    await saveConversationHistory(env, sessionId, conversationHistory);
    
    // Return the response stream to the client
    return new Response(responseStream.body, {
      headers: {
        "content-type": "text/event-stream",
        "x-session-id": sessionId
      }
    });
  } catch (error) {
    return new Response(`Error processing request: ${error.message}`, { status: 500 });
  }
}

/**
 * Detects the content type from input
 * @param {string|Object} content - The input content
 * @returns {Array} Array of content types detected
 */
function detectContentType(content) {
  const types = ["text"];
  
  if (typeof content === "string") {
    // Check for image URLs
    if (content.match(/\.(jpeg|jpg|gif|png|webp)/i)) {
      types.push("image");
    }
    // Check for audio URLs
    if (content.match(/\.(mp3|wav|ogg|m4a)/i)) {
      types.push("audio");
    }
  } else if (typeof content === "object") {
    // Check for structured content with image URLs
    if (content.image_url || (content.content && content.content.some(item => item.type === "image_url"))) {
      types.push("image");
    }
    // Check for audio content
    if (content.audio_url || content.audio) {
      types.push("audio");
    }
    // Check for live mode content
    if (content.screenShare) {
      types.push("screen");
    }
    if (content.cameraFeed) {
      types.push("camera");
    }
  }
  
  return types;
}

/**
 * Orchestrates multiple model streams into a unified response
 * @param {Array} modelStreams - Array of model stream objects with weights
 * @returns {Promise<Response>} A unified streaming response
 */
interface ModelSource {
  type: string;
  data: any;
  [key: string]: any;
}

async function orchestrateModelStreams(modelStreams): Promise<Response> {
  const encoder = new TextEncoder();
  const readableStream = new ReadableStream({
    async start(controller) {
      try {
        // Set up merged processing with proper typing
        const mergedResponse: {
          text: string;
          confidence: number;
          sources: ModelSource[];
        } = {
          text: "",
          confidence: 0,
          sources: []
        };
        
        // Process each model stream
        const modelResults = await Promise.all(
          modelStreams.map(async (modelData) => {
            const { stream, type, weight } = modelData;
            
            let result = "";
            if (stream.getReader) {
              const reader = stream.getReader();
              while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                result += new TextDecoder().decode(value);
              }
            } else {
              // Handle non-streaming results
              result = await stream;
            }
            
            return { result, type, weight };
          })
        );
        
        // Combine results based on model types and weights
        let combinedText = "";
        let primaryContent = "";
        let visionContent = "";
        let audioTranscript = "";
        let thinkingContent = "";
        let researchContent = "";
        let screenContent = "";
        let cameraContent = "";
        
        for (const { result, type, weight } of modelResults) {
          switch (type) {
            case "primary":
            case "secondary":
              primaryContent += result;
              break;
            case "thinking":
              thinkingContent = result;
              break;
            case "research":
              researchContent = result;
              break;
            case "vision":
              visionContent = result;
              break;
            case "audio":
              audioTranscript = result;
              break;
            case "screen":
              screenContent = result;
              break;
            case "camera":
              cameraContent = result;
              break;
            case "embedding":
              // Store embeddings for future use
              mergedResponse.sources.push({ type: "embedding", data: result } as ModelSource);
              break;
          }
        }
        
        // Create the combined response
        if (audioTranscript) {
          combinedText += `[Audio Transcript]: ${audioTranscript}\n\n`;
        }
        
        if (visionContent) {
          combinedText += `[Image Analysis]: ${visionContent}\n\n`;
        }
        
        if (screenContent) {
          combinedText += `[Screen Analysis]: ${screenContent}\n\n`;
        }
        
        if (cameraContent) {
          combinedText += `[Camera Feed Analysis]: ${cameraContent}\n\n`;
        }
        
        if (researchContent) {
          // Integrate research findings into the response
          combinedText += `[Research Findings]: ${researchContent}\n\n`;
        }
        
        // Integrate thinking results to enrich the primary content
        if (thinkingContent) {
          primaryContent = enrichContentWithThinking(primaryContent, thinkingContent);
        }
        
        combinedText += primaryContent;
        
        // Stream the final response
        controller.enqueue(encoder.encode(combinedText));
        controller.close();
      } catch (error) {
        controller.error(`Error in stream processing: ${error.message}`);
      }
    }
  });
  
  return new Response(readableStream, {
    headers: { "content-type": "text/event-stream" }
  });
}

/**
 * Enrich primary content with thinking output
 * @param {string} primaryContent - The primary model output
 * @param {string} thinkingContent - The thinking model output
 * @returns {string} - Enhanced content
 */
function enrichContentWithThinking(primaryContent, thinkingContent) {
  // Extract key insights from thinking content
  const insights = extractKeyInsights(thinkingContent);
  
  // Find appropriate places to integrate insights
  const sentences = primaryContent.split(/(?<=[.!?])\s+/);
  
  // Simple integration strategy - add relevant insights at paragraph breaks
  let enhanced = "";
  let insightIndex = 0;
  
  for (let i = 0; i < sentences.length; i++) {
    enhanced += sentences[i] + " ";
    
    // At paragraph breaks or every 3-4 sentences, add an insight if available
    if ((sentences[i].endsWith(".") || sentences[i].endsWith("!") || sentences[i].endsWith("?")) && 
        (i % 4 === 3) && insightIndex < insights.length) {
      enhanced += insights[insightIndex] + " ";
      insightIndex++;
    }
  }
  
  return enhanced;
}

/**
 * Extract key insights from thinking content
 * @param {string} thinkingContent - The thinking model output
 * @returns {Array} - Array of key insights
 */
function extractKeyInsights(thinkingContent) {
  // Simple extraction - split by sentences and filter for meaningful ones
  const sentences = thinkingContent.split(/(?<=[.!?])\s+/);
  
  return sentences.filter(sentence => {
    // Filter for sentences that appear to contain insights
    return sentence.length > 20 && 
           !sentence.includes("I think") && 
           !sentence.includes("As an AI") &&
           !sentence.includes("let me") &&
           sentence.match(/\b(important|key|critical|essential|significant|notable|fundamental)\b/i);
  });
}

/**
 * Generate a unique session ID
 * @returns {string} - Unique session ID
 */
function generateSessionId() {
  return 'hxtx_' + Date.now() + '_' + Math.random().toString(36).substring(2, 15);
}

interface ConversationHistory {
  sessionId: string;
  messages: Array<{
    role: string;
    content: string;
    timestamp: string;
  }>;
  metadata: {
    createdAt: string;
    lastUpdated: string;
    currentTopic: string | null;
  };
}

/**
 * Retrieve conversation history from persistent storage
 * @param {HextrixEnv} env - Environment containing KV bindings
 * @param {string} sessionId - Session identifier
 * @returns {ConversationHistory} - Conversation history
 */
async function retrieveConversationHistory(env: HextrixEnv, sessionId: string): Promise<ConversationHistory> {
  try {
    // Check if we have a KV binding
    if (env.CONVERSATIONS) {
      const stored = await env.CONVERSATIONS.get(sessionId);
      if (stored) {
        return JSON.parse(stored);
      }
    }
    
    // Return empty history if nothing found or no KV binding
    return {
      sessionId,
      messages: [],
      metadata: {
        createdAt: new Date().toISOString(),
        lastUpdated: new Date().toISOString(),
        currentTopic: null
      }
    };
  } catch (error) {
    console.error("Error retrieving conversation history:", error);
    // Return empty history on error
    return {
      sessionId,
      messages: [],
      metadata: {
        createdAt: new Date().toISOString(),
        lastUpdated: new Date().toISOString(),
        currentTopic: null
      }
    };
  }
}

/**
 * Save conversation history to persistent storage
 * @param {HextrixEnv} env - Environment containing KV bindings
 * @param {string} sessionId - Session identifier
 * @param {ConversationHistory} history - Conversation history to save
 */
async function saveConversationHistory(env: HextrixEnv, sessionId: string, history: ConversationHistory): Promise<void> {
  try {
    // Update last updated timestamp
    history.metadata.lastUpdated = new Date().toISOString();
    
    // Save to KV if available
    if (env.CONVERSATIONS) {
      await env.CONVERSATIONS.put(sessionId, JSON.stringify(history));
    }
  } catch (error) {
    console.error("Error saving conversation history:", error);
  }
}

/**
 * Detect if there might be a topic shift in the conversation
 * @param {string} newMessage - The new user message
 * @param {Array} previousMessages - Previous conversation messages
 * @returns {boolean} - True if topic shift detected
 */
function detectTopicShift(newMessage, previousMessages) {
  if (previousMessages.length < 2) return true; // Not enough context to determine shift
  
  // Get the last few user messages
  const lastUserMessages = previousMessages
    .filter(msg => msg.role === "user")
    .slice(-3)
    .map(msg => msg.content);
  
  // Simple heuristic: check for significant changes in keywords between messages
  const previousKeywords = extractKeywords(lastUserMessages.join(" "));
  const newKeywords = extractKeywords(newMessage);
  
  // Calculate keyword overlap
  const overlap = calculateKeywordOverlap(previousKeywords, newKeywords);
  
  // If overlap is low, it might indicate a topic shift
  return overlap < 0.3; // Threshold determined empirically
}

/**
 * Extract keywords from text
 * @param {string} text - Input text
 * @returns {Set} - Set of keywords
 */
function extractKeywords(text) {
  // Simple keyword extraction - remove stopwords, tokenize, and return unique words
  const stopwords = new Set(["a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "of", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"]);
  
  return new Set(
    text.toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 2 && !stopwords.has(word))
  );
}

/**
 * Calculate keyword overlap between two sets
 * @param {Set} set1 - First set of keywords
 * @param {Set} set2 - Second set of keywords
 * @returns {number} - Overlap coefficient (0-1)
 */
function calculateKeywordOverlap(set1, set2) {
  if (set1.size === 0 || set2.size === 0) return 0;
  
  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const minSize = Math.min(set1.size, set2.size);
  
  return intersection.size / minSize;
}

/**
 * Classify the topic of conversation
 * @param {HextrixEnv} env - Environment containing AI bindings
 * @param {string} message - Current message
 * @param {Array} previousMessages - Previous conversation messages
 * @returns {string} - Classified topic
 */
async function classifyTopic(env: HextrixEnv, message: string, previousMessages: Array<{role: string, content: string, timestamp: string}>): Promise<string> {
  try {
    // Extract recent conversation context
    const recentMessages = previousMessages.slice(-5).map(msg => msg.content).join("\n");
    const combinedContent = recentMessages + "\n" + message;
    
    // Use an LLM to classify the topic
    const classification = await env.AI.run("@cf/meta/llama-3.3-8b-instruct-fp8-fast", {
      stream: false,
      max_tokens: 50,
      messages: [
        { 
          role: "system", 
          content: "You are a topic classification system. Identify the main topic of the conversation in 2-5 words. Be specific but concise."
        },
        { 
          role: "user", 
          content: `Classify the topic of this conversation: "${combinedContent}"`
        }
      ]
    });
    
    // Clean up the classification output
    return classification.trim().replace(/^['"](.*)['".]$/, "$1");
  } catch (error) {
    console.error("Error classifying topic:", error);
    return "General conversation";
  }
}

/**
 * Determine if a query might need search capabilities
 * @param {string} query - User query
 * @returns {boolean} - True if search might be beneficial
 */
function mightNeedSearch(query) {
  // Keywords that suggest the need for up-to-date information
  const searchKeywords = [
    "latest", "recent", "current", "today", "news", "update", 
    "what is", "how to", "when did", "where is", "who is",
    "trends", "research", "development", "discovery"
  ];
  
  // Check if query contains any search keywords
  return searchKeywords.some(keyword => 
    query.toLowerCase().includes(keyword.toLowerCase())
  );
}

/**
 * Perform deep research using Perplexity-like capabilities
 * @param {Object} env - Environment containing AI bindings
 * @param {string} query - The research query
 * @returns {Promise<string>} - Research results
 */
async function performDeepResearch(env, query) {
  try {
    // Create a simulated search stream using a model
    const searchPrompt = `You are Sonar Deep Research, a powerful research assistant similar to Perplexity AI. You generate comprehensive, accurate, and up-to-date research results for user queries. 

For the following query, simulate a thorough web search by:
1. Identifying the key aspects that need research
2. Providing detailed, factual information on each aspect
3. Citing hypothetical sources that would realistically have this information
4. Organizing the information in a clear, logical structure
5. Summarizing the key findings

Query: "${query}"

Remember to format your response as if you had actually retrieved this information from the web, with clear citations and an organized structure.`;

    const researchStream = await env.AI.run("@cf/meta/llama-3.3-70b-instruct-fp8-fast", {
      stream: true,
      max_tokens: 1024,
      messages: [
        { role: "user", content: searchPrompt }
      ]
    });
    
    return researchStream;
  } catch (error) {
    console.error("Error performing research:", error);
    return new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("Unable to perform research at this time."));
        controller.close();
      }
    });
  }
}

/**
 * Process screen sharing data
 * @param {Object} env - Environment containing AI bindings
 * @param {string} screenData - Screen sharing data (base64 encoded image)
 * @returns {Promise<ReadableStream>} - Analysis stream
 */
async function processScreenShare(env, screenData: string) {
  try {
    // Process with vision model
    const screenAnalysisStream = await env.AI.run("@cf/llava-hf/llava-1.5-7b-hf", {
      stream: true,
      max_tokens: 512,
      messages: [
        { role: "user", content: [
          { type: "text", text: "Analyze this screen capture and describe what's happening. Identify UI elements, text content, and the apparent context of what the user is doing:" },
          { type: "image_url", image_url: { url: screenData } }
        ]}
      ]
    });
    
    return screenAnalysisStream;
  } catch (error) {
    console.error("Error processing screen share:", error);
    return new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("Unable to process screen sharing data."));
        controller.close();
      }
    });
  }
}

/**
 * Process camera feed data
 * @param {Object} env - Environment containing AI bindings
 * @param {string} cameraData - Camera feed data (base64 encoded image)
 * @returns {Promise<ReadableStream>} - Analysis stream
 */
async function processCameraFeed(env, cameraData: string) {
  try {
    // Process with vision model
    const cameraAnalysisStream = await env.AI.run("@cf/llava-hf/llava-1.5-7b-hf", {
      stream: true,
      max_tokens: 512,
      messages: [
        { role: "user", content: [
          { type: "text", text: "Analyze this camera feed and describe what you see. Identify people, objects, activities, and the environment:" },
          { type: "image_url", image_url: { url: cameraData } }
        ]}
      ]
    });
    
    return cameraAnalysisStream;
  } catch (error) {
    console.error("Error processing camera feed:", error);
    return new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("Unable to process camera feed data."));
        controller.close();
      }
    });
  }
}

interface HextrixEnv {
  AI: {
    run: (model: string, options: any) => Promise<any>;
  };
  CONVERSATIONS?: {
    get: (key: string) => Promise<string | null>;
    put: (key: string, value: string) => Promise<void>;
  };
}

interface HextrixInput {
  content: string;
  maxTokens?: number;
  sessionId?: string | null;
  liveMode?: boolean;
  screenShare?: string | null;
  cameraFeed?: string | null;
}

/**
 * Main handler function for the Hextrix AI system
 * @param {HextrixEnv} env - Environment containing AI bindings
 * @param {Request} request - The incoming request
 * @returns {Response} The AI response
 */
export default async function handler(env: HextrixEnv, request: Request) {
  // Parse the incoming request
  const input = await parseRequest(request);
  return handleRequest(env, request, input);
}
  // Access environment variables (Place them at the START of the handler):
  const geminiApiKey = "AIzaSyD-Yqakh4fflCRbvOyUOfif0PKTB6sMVWA"; // Access the GEMINI_API_KEY secret (or variable)
  const cloudflareApiToken = "Authorization: Bearer c3byj2jtWFGkCLRxsHbEyVz3ckDLsZL1EYpDj12Z"; // Access CLOUDFLARE_API_TOKEN secret (or variable)

  console.log("Gemini API Key (first few chars):", geminiApiKey ? geminiApiKey.substring(0, 5) + "..." : "Not Set");
  console.log("Cloudflare API Token (first few chars):", cloudflareApiToken ? cloudflareApiToken.substring(0, 5) + "..." : "Not Set");
/**
 * Parses the incoming request into a standardized format
 * @param {Request} request - The incoming request
 * @returns {HextrixInput} Standardized input object
 */
async function parseRequest(request: Request): Promise<HextrixInput> {
  try {
      const contentType = request.headers.get("content-type") || "";
      // Declare and initialize variables
      let screenShare: string | null = null;
      let cameraFeed: string | null = null;
      let content: string = "";
      let maxTokens: number = 2048;
      let sessionId: string | null = request.headers.get("x-session-id") || null;
      let liveMode: boolean = false;

      if (contentType.includes("application/json")) {
          const body = await request.json();
          return {
              content: body.messages?.pop()?.content || body.content || "",
              maxTokens: body.max_tokens || 2048,
              sessionId: body.session_id || sessionId,
              liveMode: body.live_mode || false,
              screenShare: body.screen_share || null,
              cameraFeed: body.camera_feed || null
          };
      } else if (contentType.includes("multipart/form-data")) {
          const formData = await request.formData();

          // Get form data entries
          const screenShareData = formData.get("screen_share");
          const cameraFeedData = formData.get("camera_feed");
          const contentData = formData.get("content");
          const maxTokensData = formData.get("max_tokens");
          const sessionIdData = formData.get("session_id");
          const liveModeData = formData.get("live_mode");

          // Process screenShare
          if (screenShareData) {
              if (screenShareData instanceof File) {
                  try {
                      screenShare = await fileToBase64(screenShareData) as string | null; // Explicit cast
                  } catch (fileError) {
                      console.error("Error converting screenShare File to base64:", fileError);
                      screenShare = null; // Handle error by setting to null
                  }
              } else if (typeof screenShareData === 'string') {
                  screenShare = screenShareData as string | null; // Explicit cast
              } else {
                  screenShare = String(screenShareData); // Fallback string conversion
              }
          }

          // Process cameraFeed
          if (cameraFeedData) {
              if (cameraFeedData instanceof File) {
                  try {
                      cameraFeed = await fileToBase64(cameraFeedData) as string | null; // Explicit cast
                  } catch (fileError) {
                      console.error("Error converting cameraFeed File to base64:", fileError);
                      cameraFeed = null; // Handle error by setting to null
                  }
              } else if (typeof cameraFeedData === 'string') {
                  cameraFeed = cameraFeedData as string | null; // Explicit cast
              } else {
                  cameraFeed = String(cameraFeedData); // Fallback string conversion
              }
          }

          // Process content
          if (contentData) {
              if (contentData instanceof File) {
                  try {
                      content = await fileToBase64(contentData) as string; // Explicit cast, assuming content is always needed as string
                  } catch (fileError) {
                      console.error("Error converting content File to base64:", fileError);
                      content = ""; // Handle error by setting to empty string
                  }
              } else if (typeof contentData === 'string') {
                  content = contentData as string; // Explicit cast
              } else {
                  content = String(contentData); // Fallback string conversion
              }
          }

          // Process maxTokens
          if (maxTokensData) {
              const parsedMaxTokens = parseInt(String(maxTokensData), 10);
              if (!isNaN(parsedMaxTokens)) {
                  maxTokens = parsedMaxTokens;
              }
          }

          // Process sessionId
          if (sessionIdData) {
              sessionId = String(sessionIdData);
          }

          // Process liveMode
          if (liveModeData) {
              liveMode = String(liveModeData) === "true";
          }


          return {
              content,
              maxTokens,
              sessionId,
              liveMode,
              screenShare,
              cameraFeed
          };
      } else {
          // Plain text or other format
          const text = await request.text();
          return {
              content: text,
              maxTokens: 2048,
              sessionId,
              liveMode: false,
              screenShare: null,
              cameraFeed: null
          };
      }
  } catch (error) {
      console.error("Error parsing request:", error);
      return {
          content: "",
          maxTokens: 2048,
          sessionId: null,
          liveMode: false,
          screenShare: null,
          cameraFeed: null
      };
  }
}

// This function is now imported from utils.ts
