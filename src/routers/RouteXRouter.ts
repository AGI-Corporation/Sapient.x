import express from "express";

const router = express.Router();

// Route.X Router - Web4AGI
// Connecting Parcel Agents to MCP Tools and Corridor Services

const MCP_TOOLS_ENDPOINT = "https://mcp.agi-corp.com/v1/tools";

// In-memory message store for agent-to-agent communication
// agent_id -> array of message envelopes
const messageInboxes: Record<string, any[]> = {};

/**
 * Route discovery endpoint
 */
router.post("/route", async (req, res) => {
  const { parcel_id, destination, intent } = req.body;
  
  console.log(`[Route.X] Routing request from ${parcel_id} to ${destination} with intent: ${intent}`);
  
  // Logic to determine which MCP tool to plug in
  let recommended_tool = "spatial_query";
  if (intent.includes("trade") || intent.includes("pay")) {
    recommended_tool = "exchange_trade";
  } else if (intent.includes("loyalty") || intent.includes("stikk")) {
    recommended_tool = "loyalty_points";
  }

  res.json({
    ok: true,
    route: {
      path: [parcel_id, "corridor_alpha", destination],
      estimated_latency: "45ms",
      mcp_integration: {
        tool: recommended_tool,
        endpoint: `${MCP_TOOLS_ENDPOINT}/${recommended_tool}`
      }
    }
  });
});

/**
 * Send a message to another agent
 * POST /messages
 */
router.post("/messages", (req, res) => {
  const envelope = req.body;
  const { to } = envelope;

  if (!to) {
    return res.status(400).json({ success: false, error: "Missing 'to' field" });
  }

  console.log(`[Route.X] Messaging: ${envelope.from} -> ${to}`);

  if (!messageInboxes[to]) {
    messageInboxes[to] = [];
  }

  messageInboxes[to].push(envelope);

  res.json({ success: true, timestamp: new Date().toISOString() });
});

/**
 * Retrieve pending messages for an agent
 * GET /messages/:agent_id
 */
router.get("/messages/:agent_id", (req, res) => {
  const { agent_id } = req.params;
  const messages = messageInboxes[agent_id] || [];

  // Clear inbox after retrieval (polling pattern)
  messageInboxes[agent_id] = [];

  res.json({
    success: true,
    agent_id,
    messages
  });
});

export default router;
