import React, { useState, useEffect, useRef } from 'react';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

import styles from './RAGChatbot.module.css';

/**
 * RAG (Retrieval Augmented Generation) Chatbot Component
 * Allows users to ask questions about the Physical AI book content
 */
const RAGChatbot = ({ title = "Physical AI Assistant", intro = "Ask me anything about Physical AI!" }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Note: Document context is not used in this component to avoid hook errors on non-doc pages
  const currentDocTitle = 'Physical AI Book';

  // Function to simulate RAG search (in a real implementation, this would call an API)
  const searchDocuments = async (query) => {
    // In a real implementation, this would:
    // 1. Call a vector database or search API
    // 2. Find relevant document chunks
    // 3. Return context for the LLM
    // For now, we'll simulate with relevant content from the book

    // Simulated document search results based on common Physical AI topics
    const documentContexts = [
      {
        title: "What is Physical AI?",
        content: "Physical AI represents the convergence of artificial intelligence and physical systems, enabling machines to interact intelligently with the real world. Unlike traditional AI that operates primarily in digital spaces, Physical AI encompasses robotics, embodied intelligence, and systems that perceive, reason, and act in physical environments. Key components include sensors for perception, processors for reasoning, and actuators for action.",
        source: "/docs/physical-ai/introduction-to-physical-ai/what-is-physical-ai"
      },
      {
        title: "Humanoid Robotics",
        content: "Humanoid robots are robots with physical human-like features. They are designed to interact with human tools and environments. Key challenges in humanoid robotics include balance control, locomotion, manipulation, and human-robot interaction. Common platforms include robots like ASIMO, Atlas, and NAO.",
        source: "/docs/physical-ai/introduction-to-physical-ai/humanoid-robotics-overview"
      },
      {
        title: "ROS 2 Framework",
        content: "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. ROS 2 uses a DDS (Data Distribution Service) implementation for communication.",
        source: "/docs/physical-ai/the-robotic-nervous-system-ros2/introduction-to-ros2"
      },
      {
        title: "Gazebo Simulation",
        content: "Gazebo is a robot simulator that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It's commonly used in robotics development to test algorithms in a safe and reproducible environment before deploying on real robots.",
        source: "/docs/physical-ai/the-digital-twin-gazebo-unity/introduction-to-gazebo-simulation"
      },
      {
        title: "NVIDIA Isaac",
        content: "NVIDIA Isaac is a complete robotics platform that includes hardware, software, and simulation tools for developing and deploying AI-powered robots. Isaac Sim is NVIDIA's robotics simulation application based on NVIDIA Omniverse, providing high-fidelity simulation for training and testing AI models.",
        source: "/docs/physical-ai/the-ai-robot-brain-nvidia-isaac/introduction-to-nvidia-isaac-sim"
      }
    ];

    // Simple keyword matching for demo purposes
    const lowerQuery = query.toLowerCase();
    const relevantDocs = documentContexts.filter(doc =>
      doc.title.toLowerCase().includes(lowerQuery) ||
      doc.content.toLowerCase().includes(lowerQuery)
    );

    return relevantDocs.length > 0 ? relevantDocs : documentContexts.slice(0, 2);
  };

  // Function to generate response based on context
  const generateResponse = async (query, context) => {
    // In a real implementation, this would call an LLM API with the context
    // For now, we'll generate a response based on the context and query
    const contextText = context.map(c => c.content).join(' ');

    // Simple response generation based on query
    let response = "I found some relevant information from the Physical AI book. ";

    if (query.toLowerCase().includes('what is') || query.toLowerCase().includes('define')) {
      if (contextText.toLowerCase().includes('physical ai')) {
        response += "Physical AI represents the convergence of artificial intelligence and physical systems, enabling machines to interact intelligently with the real world. Unlike traditional AI that operates primarily in digital spaces, Physical AI encompasses robotics, embodied intelligence, and systems that perceive, reason, and act in physical environments.";
      }
    } else if (query.toLowerCase().includes('robot') || query.toLowerCase().includes('humanoid')) {
      if (contextText.toLowerCase().includes('humanoid')) {
        response += "Humanoid robots are robots with physical human-like features designed to interact with human tools and environments. Key challenges include balance control, locomotion, manipulation, and human-robot interaction.";
      }
    } else if (query.toLowerCase().includes('ros') || query.toLowerCase().includes('framework')) {
      if (contextText.toLowerCase().includes('ros')) {
        response += "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides tools, libraries, and conventions to simplify creating complex robot behavior across various platforms, using DDS for communication.";
      }
    } else {
      response += "Based on the Physical AI book content, here's what I can tell you: " + context[0].content.substring(0, 200) + "...";
    }

    response += " For more details, check out the relevant sections in the book.";

    return response;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = { id: Date.now(), text: inputValue, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Search for relevant documents
      const context = await searchDocuments(inputValue);

      // Generate response based on context
      const response = await generateResponse(inputValue, context);

      const botMessage = {
        id: Date.now() + 1,
        text: response,
        sender: 'bot',
        context: context.map(c => ({ title: c.title, source: c.source }))
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "Sorry, I encountered an error processing your request. Please try again.",
        sender: 'bot',
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when chat is expanded
  useEffect(() => {
    if (isExpanded && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isExpanded]);

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className={styles.chatbotContainer}>
      <div className={`${styles.chatbotHeader} ${isExpanded ? styles.expanded : ''}`} onClick={toggleExpand}>
        <div className={styles.chatbotTitle}>
          <span className={styles.botIcon}>🤖</span>
          {title}
        </div>
        <div className={styles.expandIcon}>
          {isExpanded ? '−' : '+'}
        </div>
      </div>

      {isExpanded && (
        <div className={styles.chatbotBody}>
          <div className={styles.chatbotIntro}>
            {intro}
          </div>

          <div className={styles.messagesContainer}>
            {messages.length === 0 ? (
              <div className={styles.welcomeMessage}>
                <p>Hello! I'm your Physical AI learning assistant.</p>
                <p>Ask me questions about:</p>
                <ul>
                  <li>Physical AI concepts</li>
                  <li>Humanoid robotics</li>
                  <li>ROS 2 framework</li>
                  <li>Simulation tools (Gazebo, Unity)</li>
                  <li>NVIDIA Isaac platform</li>
                  <li>Vision-Language-Action systems</li>
                </ul>
                <p>I'll provide answers based on the Physical AI book content.</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`${styles.message} ${styles[message.sender]} ${message.isError ? styles.error : ''}`}
                >
                  <div className={styles.messageContent}>
                    {message.sender === 'bot' && message.context && (
                      <div className={styles.contextSources}>
                        {message.context.slice(0, 2).map((ctx, idx) => (
                          <span key={idx} className={styles.contextTag}>
                            {ctx.title}
                          </span>
                        ))}
                      </div>
                    )}
                    <div className={styles.messageText}>{message.text}</div>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className={`${styles.message} ${styles.bot}`}>
                <div className={styles.messageContent}>
                  <div className={styles.typingIndicator}>
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className={styles.inputForm}>
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask a question about Physical AI..."
              className={styles.inputField}
              disabled={isLoading}
            />
            <button
              type="submit"
              className={styles.submitButton}
              disabled={isLoading || !inputValue.trim()}
            >
              {isLoading ? '...' : '→'}
            </button>
          </form>
        </div>
      )}
    </div>
  );
};

export default RAGChatbot;