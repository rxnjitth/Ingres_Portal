import React, { useState, useEffect, useRef } from 'react';
import { GoogleGenerativeAI } from '@google/generative-ai';

const API_KEY = "AIzaSyD67ytyDvb8HA4AB-NoqizdPJ9K_kzBlzo";

const genAI = new GoogleGenerativeAI(API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-preview-05-20" });

// Professional SVG Icons
const SendIcon = ({ size = 20, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="m22 2-7 20-4-9-9-4 20-7z"/>
  </svg>
);

const MenuIcon = ({ size = 20, className = "", isHovered = false }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <line 
      x1="3" y1="6" x2="21" y2="6" 
      className={`transition-all duration-300 ${isHovered ? 'transform translate-y-0.5' : ''}`}
    />
    <line 
      x1="3" y1="12" x2="21" y2="12" 
      className={`transition-all duration-300 ${isHovered ? 'transform scale-110' : ''}`}
    />
    <line 
      x1="3" y1="18" x2="21" y2="18" 
      className={`transition-all duration-300 ${isHovered ? 'transform -translate-y-0.5' : ''}`}
    />
  </svg>
);

const CloseIcon = ({ size = 20, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="m18 6-12 12"/>
    <path d="m6 6 12 12"/>
  </svg>
);

const MoonIcon = ({ size = 20, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>
  </svg>
);

const SunIcon = ({ size = 20, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <circle cx="12" cy="12" r="4"/>
    <path d="M12 2v2"/>
    <path d="M12 20v2"/>
    <path d="m4.93 4.93 1.41 1.41"/>
    <path d="m17.66 17.66 1.41 1.41"/>
    <path d="M2 12h2"/>
    <path d="M20 12h2"/>
    <path d="m6.34 17.66-1.41-1.41"/>
    <path d="m19.07 4.93-1.41 1.41"/>
  </svg>
);

const ChatBotIcon = ({ size = 24, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>
  </svg>
);

const UserIcon = ({ size = 24, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
    <circle cx="12" cy="7" r="4"/>
  </svg>
);

const ChevronDownIcon = ({ size = 16, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="m6 9 6 6 6-6"/>
  </svg>
);

const GlobeIcon = ({ size = 16, className = "" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <circle cx="12" cy="12" r="10"/>
    <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>
    <path d="M2 12h20"/>
  </svg>
);

// Loading Animation Component
const LoadingDots = () => (
  <div className="flex items-center space-x-1">
    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
  </div>
);

// Professional Language Selector
const LanguageSelector = ({ language, setLanguage, theme }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  const languageOptions = {
    en: { name: 'English', flag: '🇬🇧' },
    hi: { name: 'हिन्दी', flag: '🇮🇳' },
    bn: { name: 'বাংলা', flag: '🇧🇩' },
    ta: { name: 'தமிழ்', flag: '🇮🇳' },
    te: { name: 'తెలుగు', flag: '🇮🇳' },
    kn: { name: 'ಕನ್ನಡ', flag: '🇮🇳' },
    ml: { name: 'മലയാളം', flag: '🇮🇳' },
    gu: { name: 'ગુજરાતી', flag: '🇮🇳' },
    mr: { name: 'मराठी', flag: '🇮🇳' },
    pa: { name: 'ਪੰਜਾਬੀ', flag: '🇮🇳' },
    ur: { name: 'اردو', flag: '🇵🇰' },
    as: { name: 'অসমীয়া', flag: '🇮🇳' },
    or: { name: 'ଓଡ଼ିଆ', flag: '🇮🇳' },
    kok: { name: 'कोंकणी', flag: '🇮🇳' }
  };

  const currentTheme = theme === 'dark' ? 
    'bg-gray-800 border-gray-600 text-white' : 
    'bg-white border-gray-300 text-gray-900';

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center space-x-2 px-3 py-2 rounded-lg border ${currentTheme} hover:bg-opacity-80 transition-all duration-300 min-w-[120px] transform hover:scale-105`}
      >
        <GlobeIcon size={16} />
        <span className="text-sm font-medium">{languageOptions[language].name}</span>
        <ChevronDownIcon size={14} className={`transform transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      
      {isOpen && (
        <div className={`absolute top-full left-0 mt-2 w-48 rounded-xl shadow-2xl border ${currentTheme} z-50 max-h-60 overflow-y-auto backdrop-blur-lg bg-opacity-95`}>
          {Object.entries(languageOptions).map(([code, { name, flag }]) => (
            <button
              key={code}
              onClick={() => {
                setLanguage(code);
                setIsOpen(false);
              }}
              className={`w-full flex items-center space-x-3 px-4 py-3 text-left hover:bg-blue-50 dark:hover:bg-gray-700 transition-all duration-200 first:rounded-t-xl last:rounded-b-xl ${language === code ? 'bg-blue-100 dark:bg-gray-700' : ''}`}
            >
              <span className="text-lg">{flag}</span>
              <span className="text-sm font-medium">{name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [theme, setTheme] = useState('light');
  const [language, setLanguage] = useState('en');
  const [menuHovered, setMenuHovered] = useState(false);
  
  const messagesEndRef = useRef(null);

  const themeStyles = {
    light: {
      background: 'bg-gray-50',
      text: 'text-gray-900',
      headerBg: 'bg-white',
      chatBg: 'bg-white',
      messageBg: 'bg-gray-100',
      userMessageBg: 'bg-blue-600',
      inputBg: 'bg-white',
      inputBorder: 'border-gray-300',
      inputPlaceholder: 'placeholder-gray-500',
      sidebarBg: 'bg-white',
      borderColor: 'border-gray-200'
    },
    dark: {
      background: 'bg-gray-900',
      text: 'text-gray-100',
      headerBg: 'bg-gray-800',
      chatBg: 'bg-gray-900',
      messageBg: 'bg-gray-800',
      userMessageBg: 'bg-blue-600',
      inputBg: 'bg-gray-800',
      inputBorder: 'border-gray-600',
      inputPlaceholder: 'placeholder-gray-400',
      sidebarBg: 'bg-gray-800',
      borderColor: 'border-gray-700'
    }
  };

  const languageOptions = {
    en: 'English', hi: 'हिन्दी', bn: 'বাংলা', ta: 'தமிழ்', te: 'తెలుగు',
    kn: 'ಕನ್ನಡ', ml: 'മലയാളം', gu: 'ગুજરાતી', mr: 'मराठী', pa: 'ਪੰਜਾਬੀ',
    ur: 'اردو', as: 'অসমীয়া', or: 'ଓଡ଼ିଆ', kok: 'कोंकणी'
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
      const prompt = `You are INGIN, a professional AI assistant for the INGRES (Indian National Groundwater Resource Estimation System) portal. Respond to this query in ${languageOptions[language]}: "${text}". Provide accurate, comprehensive information about groundwater resources, assessments, and related data. Keep responses professional, helpful, and conversational.`;
      
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
        text: language === 'hi' ? "क्षमा करें, तकनीकी समस्या के कारण मैं आपकी सहायता नहीं कर सकता। कृपया पुनः प्रयास करें।" : "I apologize for the technical issue. Please try again.", 
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

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const resetChat = () => {
    setMessages([]);
    setIsSidebarOpen(false);
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className={`h-screen flex ${currentTheme.background} ${currentTheme.text} transition-all duration-500 overflow-hidden`}>
      {/* Sidebar Overlay for Mobile */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden transition-opacity duration-300"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Fixed Sidebar */}
      <div className={`fixed left-0 top-0 h-screen w-80 ${currentTheme.sidebarBg} ${currentTheme.borderColor} border-r shadow-2xl transform transition-all duration-500 ease-in-out z-50 ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0`}>
        <div className="flex flex-col h-full">
          {/* Sidebar Header */}
          <div className={`flex items-center justify-between p-6 ${currentTheme.borderColor} border-b shrink-0`}>
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl flex items-center justify-center shadow-xl transform transition-transform duration-300 hover:scale-110">
                <ChatBotIcon size={20} className="text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">INGIN</h2>
                <p className="text-xs text-gray-500 dark:text-gray-400">AI Assistant</p>
              </div>
            </div>
            <button
              onClick={() => setIsSidebarOpen(false)}
              className="lg:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-300 transform hover:scale-110"
            >
              <CloseIcon size={18} />
            </button>
          </div>

          {/* Quick Actions */}
          <div className="p-6 space-y-4 shrink-0">
            <h3 className="font-semibold text-sm text-gray-600 dark:text-gray-400 uppercase tracking-wide">Quick Actions</h3>
            <button
              onClick={resetChat}
              className="w-full text-left p-4 rounded-xl hover:bg-blue-50 dark:hover:bg-gray-700 transition-all duration-300 group transform hover:scale-105 shadow-sm"
            >
              <span className="font-medium group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">New Conversation</span>
            </button>
          </div>

          {/* Footer */}
          <div className={`mt-auto p-6 ${currentTheme.borderColor} border-t shrink-0`}>
            <div className="text-center">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                INGRES Portal Assistant
              </div>
              <div className="w-16 h-1 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full mx-auto"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className={`flex-1 flex flex-col lg:ml-80 transition-all duration-500`}>
        {/* Fixed Header */}
        <div className={`fixed top-0 right-0 left-0 lg:left-80 ${currentTheme.headerBg} ${currentTheme.borderColor} border-b px-4 py-4 shadow-lg backdrop-blur-lg bg-opacity-95 z-40 transition-all duration-500`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                onMouseEnter={() => setMenuHovered(true)}
                onMouseLeave={() => setMenuHovered(false)}
                className="p-3 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-300 lg:hidden transform hover:scale-110 shadow-lg"
              >
                <MenuIcon size={20} isHovered={menuHovered} />
              </button>
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl flex items-center justify-center shadow-xl transform transition-transform duration-300 hover:scale-110">
                  <ChatBotIcon size={24} className="text-white" />
                </div>
                <div className="hidden sm:block">
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">INGIN Assistant</h1>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Powered by Advanced AI</p>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <LanguageSelector language={language} setLanguage={setLanguage} theme={theme} />
              <button
                onClick={toggleTheme}
                className={`p-3 rounded-xl ${currentTheme.inputBg} ${currentTheme.borderColor} border hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-300 shadow-lg transform hover:scale-110`}
                title={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
              >
                <div className="transform transition-transform duration-300 hover:rotate-180">
                  {theme === 'light' ? <MoonIcon size={18} /> : <SunIcon size={18} />}
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Scrollable Chat Area with proper spacing */}
        <div className="flex-1 flex flex-col pt-20 pb-24">
          {/* Chat Messages Container */}
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            <div className="p-6">
              {messages.length === 0 ? (
                <div className="flex items-center justify-center min-h-full">
                  <div className="text-center px-6 max-w-2xl">
                    <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-blue-600 to-purple-600 rounded-full flex items-center justify-center shadow-2xl transform transition-transform duration-300 hover:scale-110">
                      <ChatBotIcon size={32} className="text-white" />
                    </div>
                    <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Welcome to INGIN</h1>
                    <p className="text-lg text-gray-600 dark:text-gray-300 leading-relaxed">
                      Your intelligent assistant for the Indian National Groundwater Resource Estimation System. 
                      I'm here to help you explore groundwater data, understand assessments, and navigate portal features with precision and expertise.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-6 max-w-4xl mx-auto">
                  {messages.map((msg, index) => (
                    <div
                      key={index}
                      className={`flex items-start space-x-4 ${msg.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''} transform transition-all duration-300 hover:scale-[1.02]`}
                    >
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center shadow-xl ${
                        msg.sender === 'user' 
                          ? 'bg-gradient-to-br from-blue-600 to-blue-700' 
                          : 'bg-gradient-to-br from-gray-600 to-gray-700'
                      } transform transition-transform duration-300 hover:scale-110`}>
                        {msg.sender === 'user' ? 
                          <UserIcon size={18} className="text-white" /> : 
                          <ChatBotIcon size={18} className="text-white" />
                        }
                      </div>
                      <div className={`max-w-3xl p-5 rounded-2xl shadow-lg transition-all duration-300 hover:shadow-xl ${
                        msg.sender === 'user'
                          ? `${currentTheme.userMessageBg} text-white rounded-tr-sm`
                          : `${currentTheme.messageBg} ${currentTheme.text} rounded-tl-sm`
                      }`}>
                        <div className="whitespace-pre-wrap leading-relaxed">{msg.text}</div>
                        <div className="text-xs mt-3 opacity-70">
                          {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {isLoading && (
                    <div className="flex items-start space-x-4">
                      <div className="w-10 h-10 bg-gradient-to-br from-gray-600 to-gray-700 rounded-full flex items-center justify-center shadow-xl">
                        <ChatBotIcon size={18} className="text-white" />
                      </div>
                      <div className={`${currentTheme.messageBg} ${currentTheme.text} p-5 rounded-2xl rounded-tl-sm shadow-lg`}>
                        <LoadingDots />
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Fixed Input Area */}
        <div className={`fixed bottom-0 right-0 left-0 lg:left-80 ${currentTheme.borderColor} border-t backdrop-blur-lg bg-opacity-95 p-6 z-40 transition-all duration-500`}>
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
            <div className={`flex items-center space-x-4 ${currentTheme.inputBg} ${currentTheme.borderColor} border rounded-2xl shadow-2xl focus-within:ring-4 focus-within:ring-blue-500 focus-within:ring-opacity-30 focus-within:border-transparent transition-all duration-300 transform focus-within:scale-[1.02]`}>
              <input
                type="text"
                placeholder="Ask me about groundwater resources, assessments, or portal features..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                className={`flex-1 px-6 py-5 bg-transparent ${currentTheme.text} ${currentTheme.inputPlaceholder} focus:outline-none text-lg`}
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="mr-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-xl p-4 hover:from-blue-700 hover:to-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-xl hover:shadow-2xl transform hover:scale-110 disabled:hover:scale-100"
              >
                <SendIcon size={20} />
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
