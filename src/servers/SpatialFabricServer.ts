import express from "express";
import fs from "fs";
import path from "path";

const app = express();
app.use(express.json());

const DATA = path.join(__dirname, "../data");

// GET /placeHierarchy
app.get("/placeHierarchy", (req, res) => {
  const { lat, lon } = req.query;
  res.json({
    country: "us",
    region: "ca",
    county: "san-francisco",
    city: "san-francisco",
    parcel_id: `us-ca-sf-${Math.abs(Math.round(Number(lat) * 1000))}-${Math.abs(Math.round(Number(lon) * 1000))}`
  });
});

// GET /parcel/:parcel_id
app.get("/parcel/:parcel_id", (req, res) => {
  const parcels = JSON.parse(fs.readFileSync(`${DATA}/sf-parcels.geojson`, "utf8"));
  const feature = parcels.features.find(
    (f: any) => f.properties.parcel_id === req.params.parcel_id
  ) || parcels.features[0];
  res.json(feature);
});

// GET /layers
app.get("/layers", (_req, res) => {
  res.json({
    layers: [
      { id: "parcels", name: "Parcels", type: "polygon" },
      { id: "utilities-underground", name: "Underground Utilities", type: "line" },
      { id: "stikk_spots", name: "Stikk Spots", type: "point" },
      { id: "routes", name: "Routes", type: "line" }
    ]
  });
});

app.listen(3001, () => console.log("SpatialFabricServer on :3001"));
