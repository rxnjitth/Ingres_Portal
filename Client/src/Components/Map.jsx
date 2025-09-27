import React, { useState } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// Mock GeoJSON data for a conceptual Tamil Nadu map with a few districts.
// In a real application, you would load a proper GeoJSON file here.
const TAMIL_NADU_GEOJSON = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": { "name": "Coimbatore", "waterLevel": "Safe", "latitude": "11.0045° N", "longitude": "76.9616° E", "waterLevelStatus": "Normal" },
      "geometry": { "type": "Polygon", "coordinates": [[ [76.5, 11.2], [77.5, 11.2], [77.5, 10.5], [76.5, 10.5], [76.5, 11.2] ]] }
    },
    {
      "type": "Feature",
      "properties": { "name": "Chennai", "waterLevel": "Critical", "latitude": "13.0827° N", "longitude": "80.2707° E", "waterLevelStatus": "Critical" },
      "geometry": { "type": "Polygon", "coordinates": [[ [80.2, 13.1], [80.4, 13.1], [80.4, 12.9], [80.2, 12.9], [80.2, 13.1] ]] }
    },
    {
      "type": "Feature",
      "properties": { "name": "Madurai", "waterLevel": "Over-Exploited", "latitude": "9.9252° N", "longitude": "78.1198° E", "waterLevelStatus": "Over-Exploited" },
      "geometry": { "type": "Polygon", "coordinates": [[ [78.0, 10.0], [78.5, 10.0], [78.5, 9.5], [78.0, 9.5], [78.0, 10.0] ]] }
    },
    {
      "type": "Feature",
      "properties": { "name": "Salem", "waterLevel": "Semi-Critical", "latitude": "11.6643° N", "longitude": "78.1481° E", "waterLevelStatus": "Semi-Critical" },
      "geometry": { "type": "Polygon", "coordinates": [[ [77.9, 11.7], [78.3, 11.7], [78.3, 11.5], [77.9, 11.5], [77.9, 11.7] ]] }
    }
  ]
};

// Mock data for the overall summary.
const MOCK_DATA_SUMMARY = {
  total: 6737,
  safe: 62337,
  semiCritical: 7293,
  critical: 67347,
  overExploited: 637327,
  saline: 6737
};

// Function to get color based on water level category.
const getColor = (level) => {
  switch (level.toLowerCase()) {
    case 'safe':
    case 'normal':
      return '#3b82f6'; // blue
    case 'semi-critical':
      return '#f97316'; // orange
    case 'critical':
      return '#eab308'; // yellow
    case 'over-exploited':
    case 'exploited':
      return '#ef4444'; // red
    case 'saline':
      return '#22c55e'; // green
    default:
      return '#94a3b8'; // gray
  }
};

// Component to display the data legend on the left.
const DataLegend = ({ data }) => {
  const categories = [
    { name: 'Total', count: data.total, color: '#94a3b8' },
    { name: 'Safe', count: data.safe, color: '#3b82f6' },
    { name: 'Semi-Critical', count: data.semiCritical, color: '#f97316' },
    { name: 'Critical', count: data.critical, color: '#eab308' },
    { name: 'Over-Exploited', count: data.overExploited, color: '#ef4444' },
    { name: 'Saline', count: data.saline, color: '#22c55e' }
  ];

  return (
    <div className="absolute left-6 top-1/2 transform -translate-y-1/2 bg-white rounded-3xl shadow-lg border border-slate-200 p-4 flex flex-col space-y-4 z-50">
      <h3 className="font-bold text-lg text-gray-900">Legend</h3>
      {categories.map((category) => (
        <div key={category.name} className="flex items-center space-x-3">
          <div className="w-4 h-4 rounded-full" style={{ backgroundColor: category.color }}></div>
          <span className="text-sm font-semibold text-gray-600">{category.name} ({category.count})</span>
        </div>
      ))}
    </div>
  );
};

// Main Map component.
export default function Map() {
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);

  const onEachFeature = (feature, layer) => {
    const props = feature.properties;
    layer.bindPopup(`
      <div style="font-family: sans-serif; font-size: 14px;">
        <h3 style="margin: 0 0 8px 0; font-weight: bold;">${props.name} Details</h3>
        <p><strong>Area:</strong> ${props.name}</p>
        <p><strong>Latitude:</strong> ${props.latitude}</p>
        <p><strong>Longitude:</strong> ${props.longitude}</p>
        <p><strong>Water Level:</strong> <span style="background-color: ${getColor(props.waterLevel)}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold;">${props.waterLevelStatus}</span></p>
      </div>
    `);
  };

  // Toggle chatbot visibility.
  const toggleChatbot = () => {
    setIsChatbotOpen(!isChatbotOpen);
  };

  return (
    <div className="flex h-screen w-full bg-slate-100 font-sans text-gray-800 transition-all duration-300">
      {/* Main Map Content Area */}
      <div className="relative flex-1 p-6 flex items-center justify-center">
        {/* Map Container - Leaflet map goes here */}
        <div className="w-full h-full bg-white rounded-3xl shadow-2xl overflow-hidden">
          <MapContainer center={[11.1271, 78.6569]} zoom={7} style={{ height: '100%', width: '100%' }}>
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            <GeoJSON
              data={TAMIL_NADU_GEOJSON}
              style={(feature) => ({
                fillColor: getColor(feature.properties.waterLevel),
                weight: 1,
                opacity: 1,
                color: 'white',
                dashArray: '3',
                fillOpacity: 0.7
              })}
              onEachFeature={onEachFeature}
            />
          </MapContainer>
        </div>
      </div>

      {/* Data Legend */}
      <DataLegend data={MOCK_DATA_SUMMARY} />
      
      {/* Chatbot Toggle Icon */}
      <button
        onClick={toggleChatbot}
        className="fixed bottom-6 right-6 p-4 bg-blue-600 text-white rounded-full shadow-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 z-50 transition-transform transform hover:scale-110"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 24 24" fill="currentColor">
          <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z" />
        </svg>
      </button>

      {/* Right Sidebar for Chatbot */}
      {isChatbotOpen && (
        <div className="w-1/4 p-6 bg-slate-50 flex flex-col justify-end items-center shadow-xl">
          <div className="w-full flex-grow flex items-center justify-center">
            <div className="bg-white p-6 rounded-3xl shadow-inner border border-gray-200 text-center w-full">
              <h2 className="text-xl font-bold mb-2 text-gray-800">Ask Your Doubt! I'll Help You</h2>
              <p className="text-sm text-gray-500">Chat with the INGRES AI assistant.</p>
              <div className="mt-4 p-4 rounded-lg bg-blue-50 text-blue-800 text-sm">
                <p>Chatbot UI placeholder.</p>
                <p>Integrate your `Chatbot.jsx` here.</p>
              </div>
            </div>
          </div>
          <div className="w-full mt-4">
            <input
              type="text"
              className="w-full p-3 rounded-full border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Chat with INGIN"
            />
          </div>
        </div>
      )}
    </div>
  );
}


