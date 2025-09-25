import React, { useState, useEffect, useRef } from 'react';
import { GoogleGenerativeAI } from '@google/generative-ai';

const API_KEY = "";

const genAI = new GoogleGenerativeAI(API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-preview-05-20" });

// Professional SVG Icons
const SendIcon = ({ size = 20, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <line x1="22" y1="2" x2="11" y2="13"></line>
    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
  </svg>
);

const MenuIcon = ({ size = 24, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <line x1="3" y1="12" x2="21" y2="12"></line>
    <line x1="3" y1="6" x2="21" y2="6"></line>
    <line x1="3" y1="18" x2="21" y2="18"></line>
  </svg>
);

const ChevronLeftIcon = ({ size = 20, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <polyline points="15 18 9 12 15 6"></polyline>
  </svg>
);

const CloseIcon = ({ size = 24, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);

const FileIcon = ({ size = 20, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
    <polyline points="14 2 14 8 20 8"></polyline>
    <line x1="16" y1="13" x2="8" y2="13"></line>
    <line x1="16" y1="17" x2="8" y2="17"></line>
  </svg>
);

const PortalIcon = ({ size = 20, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="6" x2="12" y2="12"></line>
    <line x1="16" y1="12" x2="12" y2="12"></line>
  </svg>
);

const ChatIcon = ({ size = 48, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
  </svg>
);

const LoadingDots = () => (
  <div className="flex items-center space-x-1">
    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
  </div>
);

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [theme, setTheme] = useState('light');
  const [language, setLanguage] = useState('en');
  
  const messagesEndRef = useRef(null);

  const themeStyles = {
    light: {
      background: 'bg-gray-50',
      text: 'text-gray-900',
      headerBg: 'bg-white',
      chatBg: 'bg-white',
      messageBg: 'bg-gray-100',
      inputBg: 'bg-white',
      inputBorder: 'border-gray-200',
      inputPlaceholder: 'placeholder-gray-500',
      cardBg: 'bg-white',
      hoverBg: 'hover:bg-gray-50'
    },
    dark: {
      background: 'bg-gray-900',
      text: 'text-gray-100',
      headerBg: 'bg-gray-800',
      chatBg: 'bg-gray-800',
      messageBg: 'bg-gray-700',
      inputBg: 'bg-gray-700',
      inputBorder: 'border-gray-600',
      inputPlaceholder: 'placeholder-gray-400',
      cardBg: 'bg-gray-700',
      hoverBg: 'hover:bg-gray-600'
    }
  };

  const predefinedQueries = [
    "What are the key findings of the latest groundwater assessment?",
    "Show me historical groundwater data for the year 2022.",
    "Explain the 'Safe' and 'Over-Exploited' categories.",
    "What is the annual groundwater recharge in my region?",
    "Provide an overview of the INGRES portal."
  ];
  
  const languageOptions = {
    en: 'English',
    hi: 'हिन्दी',
    bn: 'বাংলা',
    ta: 'தமிழ்',
    te: 'తెలుగు',
    kn: 'ಕನ್ನಡ',
    ml: 'മലയാളം',
    gu: 'ગુજરાતી',
    mr: 'मराठी',
    pa: 'ਪੰਜਾਬੀ',
    ur: 'اردو',
    as: 'অসমীয়া',
    or: 'ଓଡ଼ିଆ',
    kok: 'कोंकणी'
  };

  const currentTheme = themeStyles[theme];

  const handleSendMessage = async (messageText) => {
    const text = messageText || input.trim();
    if (text === '') return;

    const userMessage = { text, sender: 'user', timestamp: new Date() };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const prompt = `Act as a professional virtual assistant for the INGRES (Indian National Groundwater Resource Estimation System) portal. The user asks the following question in ${languageOptions[language]}: "${text}". Please provide a comprehensive, accurate, and helpful response in the same language. Focus on groundwater data, assessments, and related information.`;
      
      const result = await model.generateContent({
        contents: [{ parts: [{ text: prompt }] }],
        tools: [{ google_search: {} }]
      });

      const responseText = result.response.text();
      const botMessage = { text: responseText, sender: 'bot', timestamp: new Date() };
      setMessages((prevMessages) => [...prevMessages, botMessage]);

    } catch (error) {
      console.error("Error generating response:", error);
      const errorMessage = { 
        text: language === 'hi' ? "क्षमा करें, मैं अभी आपके अनुरोध को प्रोसेस करने में असमर्थ हूं। कृपया बाद में पुनः प्रयास करें।" : "I apologize, but I'm unable to process your request at this time. Please try again later.", 
        sender: 'bot', 
        timestamp: new Date() 
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    handleSendMessage();
  };

  const handlePredefinedQuery = (query) => {
    handleSendMessage(query);
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className={`flex h-screen font-sans ${currentTheme.background} ${currentTheme.text} transition-colors duration-300`}>
      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 transform ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:relative lg:translate-x-0 transition-transform duration-300 ease-in-out z-50 w-72 bg-slate-900 text-white shadow-2xl flex flex-col`}>
        <div className="flex items-center justify-between p-6 border-b border-slate-700">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <PortalIcon size={18} className="text-white" />
            </div>
            <h2 className="text-xl font-bold text-white">INGIN</h2>
          </div>
          <button
            onClick={() => setIsSidebarOpen(false)}
            className="lg:hidden text-gray-300 hover:text-white transition-colors"
          >
            <CloseIcon size={20} />
          </button>
        </div>
        
        <nav className="flex-1 px-4 py-6 space-y-2">
          <a
            href="#"
            className="flex items-center px-4 py-3 text-gray-300 transition-colors duration-200 rounded-lg hover:bg-slate-800 hover:text-white group"
          >
            <ChevronLeftIcon size={18} className="mr-3 group-hover:text-blue-400" />
            <span className="font-medium">Go to Portal</span>
          </a>
          <a
            href="#"
            className="flex items-center px-4 py-3 text-gray-300 transition-colors duration-200 rounded-lg hover:bg-slate-800 hover:text-white group"
          >
            <FileIcon size={18} className="mr-3 group-hover:text-blue-400" />
            <span className="font-medium">Download Report</span>
          </a>
          <a
            href="#"
            className="flex items-center px-4 py-3 text-gray-300 transition-colors duration-200 rounded-lg hover:bg-slate-800 hover:text-white group"
          >
            <FileIcon size={18} className="mr-3 group-hover:text-blue-400" />
            <span className="font-medium">Documentation</span>
          </a>
        </nav>
        
        <div className="p-4 border-t border-slate-700">
          <div className="text-xs text-gray-400 text-center">
            INGRES Portal Assistant
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className={`flex-1 flex flex-col ${currentTheme.chatBg} transition-colors duration-300`}>
        {/* Header */}
        <div className={`flex items-center justify-between p-4 ${currentTheme.headerBg} border-b shadow-sm transition-colors duration-300`}>
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setIsSidebarOpen(true)}
              className={`${currentTheme.text} lg:hidden hover:bg-gray-100 dark:hover:bg-gray-700 p-2 rounded-lg transition-colors`}
            >
              <MenuIcon size={20} />
            </button>
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg flex items-center justify-center shadow-lg">
                <ChatIcon size={20} className="text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-blue-600">INGIN Assistant</h1>
                <p className="text-sm text-gray-500 dark:text-gray-400">INGRES Portal Chatbot</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className={`px-3 py-2 text-sm ${currentTheme.inputBg} ${currentTheme.text} ${currentTheme.inputBorder} border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors`}
            >
              {Object.entries(languageOptions).map(([code, name]) => (
                <option key={code} value={code}>{name}</option>
              ))}
            </select>
            
            <select
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              className={`px-3 py-2 text-sm ${currentTheme.inputBg} ${currentTheme.text} ${currentTheme.inputBorder} border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors`}
            >
              <option value="light">Light Theme</option>
              <option value="dark">Dark Theme</option>
            </select>
          </div>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center max-w-4xl mx-auto">
              <div className="w-24 h-24 mb-6 flex items-center justify-center bg-gradient-to-br from-blue-100 to-blue-200 dark:from-blue-900 dark:to-blue-800 rounded-full shadow-lg">
                <ChatIcon size={48} className="text-blue-600 dark:text-blue-400" />
              </div>
              <h2 className={`text-3xl font-bold ${currentTheme.text} mb-4`}>Welcome to INGIN</h2>
              <p className="text-gray-600 dark:text-gray-300 max-w-2xl mb-8 text-lg leading-relaxed">
                I'm your intelligent assistant for the INGRES portal. I can help you with groundwater data analysis, 
                historical assessments, resource categorization, and portal navigation. Ask me anything about 
                India's groundwater resources.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-4xl">
                {predefinedQueries.map((query, index) => (
                  <button
                    key={index}
                    onClick={() => handlePredefinedQuery(query)}
                    className={`${currentTheme.cardBg} ${currentTheme.text} text-left p-4 rounded-xl shadow-md ${currentTheme.hoverBg} transition-all duration-200 hover:shadow-lg border border-gray-200 dark:border-gray-600 hover:border-blue-300 dark:hover:border-blue-500`}
                  >
                    <div className="flex items-start space-x-3">
                      <div className="w-6 h-6 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                        <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                      </div>
                      <span className="text-sm font-medium leading-relaxed">{query}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6 max-w-4xl mx-auto">
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`flex items-start space-x-3 max-w-3xl ${msg.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      msg.sender === 'user' 
                        ? 'bg-blue-600' 
                        : 'bg-gray-300 dark:bg-gray-600'
                    }`}>
                      {msg.sender === 'user' ? (
                        <div className="w-4 h-4 bg-white rounded-full"></div>
                      ) : (
                        <ChatIcon size={16} className="text-gray-600 dark:text-gray-300" />
                      )}
                    </div>
                    <div
                      className={`p-4 rounded-2xl shadow-sm ${
                        msg.sender === 'user'
                          ? 'bg-blue-600 text-white rounded-tr-sm'
                          : `${currentTheme.messageBg} ${currentTheme.text} rounded-tl-sm`
                      }`}
                    >
                      <div className="whitespace-pre-wrap leading-relaxed">{msg.text}</div>
                    </div>
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="flex justify-start">
                  <div className="flex items-start space-x-3 max-w-3xl">
                    <div className="w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded-full flex items-center justify-center flex-shrink-0">
                      <ChatIcon size={16} className="text-gray-600 dark:text-gray-300" />
                    </div>
                    <div className={`${currentTheme.messageBg} ${currentTheme.text} p-4 rounded-2xl rounded-tl-sm shadow-sm`}>
                      <LoadingDots />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Message Input */}
        <div className="p-6 border-t bg-gray-50 dark:bg-gray-800 transition-colors duration-300">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={handleSubmit} className="relative">
              <div className={`flex items-center ${currentTheme.inputBg} rounded-2xl shadow-lg border ${currentTheme.inputBorder} focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-transparent transition-all duration-200`}>
                <input
                  type="text"
                  placeholder="Ask me about groundwater data, assessments, or portal features..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  className={`flex-1 px-6 py-4 ${currentTheme.inputBg} ${currentTheme.text} ${currentTheme.inputPlaceholder} rounded-2xl focus:outline-none`}
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="mr-2 bg-blue-600 text-white rounded-xl p-3 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                >
                  <SendIcon size={20} />
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;