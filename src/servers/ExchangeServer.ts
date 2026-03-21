import express from "express";

const app = express();
app.use(express.json());

// Universal Exchange Apparatus (UEA) - Web4AGI
// Integrating USDx stablecoins and x402 protocols

interface Trade {
  id: string;
  sender_parcel: string;
  receiver_parcel: string;
  asset_symbol: string; // USDx for stablecoin
  amount: number;
  status: "pending" | "completed" | "failed";
  protocol: "x402";
  timestamp: string;
}

const trades: Record<string, Trade> = {};

app.post("/exchange/trade", (req, res) => {
  const { sender, receiver, amount, asset = "USDx" } = req.body;
  const tradeId = `tx_${Date.now()}`;

  const newTrade: Trade = {
    id: tradeId,
    sender_parcel: sender,
    receiver_parcel: receiver,
    asset_symbol: asset,
    amount,
    status: "pending",
    protocol: "x402",
    timestamp: new Date().toISOString()
  };

  trades[tradeId] = newTrade;

  // Simulate x402 protocol handshake
  console.log(`[UEA] Initiating x402 trade ${tradeId} between ${sender} and ${receiver}`);

  setTimeout(() => {
    trades[tradeId].status = "completed";
    console.log(`[UEA] Trade ${tradeId} completed successfully.`);
  }, 2000);

  res.json({ ok: true, trade_id: tradeId, status: "initiated" });
});

app.get("/exchange/status/:trade_id", (req, res) => {
  const { trade_id } = req.params;
  const trade = trades[trade_id];
  if (trade) {
    res.json(trade);
  } else {
    res.status(404).json({ error: "Trade not found" });
  }
});

const PORT = 3003;
app.listen(PORT, () => {
  console.log(`UEA Exchange Server running on port ${PORT}`);
});
