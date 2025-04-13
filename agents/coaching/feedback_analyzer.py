"""Natural language feedback analyzer for agent optimization."""

from typing import Dict, List, Tuple
from collections import defaultdict
import spacy
from textblob import TextBlob
from dataclasses import dataclass
import json

@dataclass
class FeedbackItem:
    """A single piece of feedback."""
    text: str
    agent_id: str
    timestamp: float
    user_id: str
    context: Dict
    
@dataclass
class FeedbackAnalysis:
    """Analysis of feedback items."""
    sentiment_score: float
    topics: List[str]
    suggestions: List[str]
    common_issues: List[Tuple[str, int]]
    
class FeedbackAnalyzer:
    """Analyzes natural language feedback to improve agent performance."""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.feedback_items: List[FeedbackItem] = []
        self.topic_patterns = {
            "response_time": ["slow", "fast", "quick", "delay", "wait"],
            "accuracy": ["wrong", "incorrect", "accurate", "precise", "error"],
            "clarity": ["clear", "unclear", "confusing", "understand", "vague"],
            "helpfulness": ["helpful", "unhelpful", "useful", "useless"],
            "personality": ["friendly", "rude", "professional", "tone"]
        }
        
    def add_feedback(self, feedback: FeedbackItem) -> None:
        """Add a feedback item for analysis."""
        self.feedback_items.append(feedback)
        
    def analyze_feedback(self, agent_id: str) -> FeedbackAnalysis:
        """Analyze feedback for a specific agent."""
        # Get relevant feedback
        agent_feedback = [
            f for f in self.feedback_items 
            if f.agent_id == agent_id
        ]
        
        if not agent_feedback:
            raise ValueError(f"No feedback found for agent {agent_id}")
            
        # Analyze sentiment
        sentiment_scores = []
        for feedback in agent_feedback:
            blob = TextBlob(feedback.text)
            sentiment_scores.append(blob.sentiment.polarity)
            
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Extract topics and issues
        topics = defaultdict(int)
        issues = defaultdict(int)
        
        for feedback in agent_feedback:
            doc = self.nlp(feedback.text.lower())
            
            # Check for topic patterns
            for topic, patterns in self.topic_patterns.items():
                if any(pattern in feedback.text.lower() for pattern in patterns):
                    topics[topic] += 1
                    
            # Extract potential issues (negative sentences)
            for sent in doc.sents:
                blob_sent = TextBlob(sent.text)
                if blob_sent.sentiment.polarity < -0.2:
                    issues[sent.text.strip()] += 1
                    
        # Generate suggestions based on analysis
        suggestions = self._generate_suggestions(
            topics, 
            issues, 
            avg_sentiment
        )
        
        return FeedbackAnalysis(
            sentiment_score=avg_sentiment,
            topics=sorted(topics, key=topics.get, reverse=True),
            suggestions=suggestions,
            common_issues=sorted(
                issues.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        )
        
    def _generate_suggestions(self, 
                            topics: Dict[str, int],
                            issues: Dict[str, int],
                            sentiment: float) -> List[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        # Suggestions based on topics
        if topics["response_time"] > 0:
            suggestions.append(
                "Optimize response generation for speed"
            )
        if topics["accuracy"] > 0:
            suggestions.append(
                "Enhance fact-checking and validation processes"
            )
        if topics["clarity"] > 0:
            suggestions.append(
                "Improve response structure and explanation clarity"
            )
        if topics["helpfulness"] > 0:
            suggestions.append(
                "Focus on providing more actionable solutions"
            )
        if topics["personality"] > 0:
            suggestions.append(
                "Adjust tone and personality settings"
            )
            
        # Suggestions based on sentiment
        if sentiment < -0.2:
            suggestions.append(
                "Review and revise core interaction patterns"
            )
        elif sentiment < 0:
            suggestions.append(
                "Fine-tune response quality and helpfulness"
            )
            
        return suggestions
    
    def generate_report(self, agent_id: str, filepath: str) -> None:
        """Generate a detailed feedback analysis report."""
        analysis = self.analyze_feedback(agent_id)
        
        report = {
            "agent_id": agent_id,
            "overall_sentiment": {
                "score": analysis.sentiment_score,
                "interpretation": "Positive" if analysis.sentiment_score > 0
                                else "Negative" if analysis.sentiment_score < 0
                                else "Neutral"
            },
            "key_topics": [
                {"topic": topic, "frequency": self.topic_patterns[topic]}
                for topic in analysis.topics
            ],
            "common_issues": [
                {"text": issue, "frequency": freq}
                for issue, freq in analysis.common_issues
            ],
            "improvement_suggestions": analysis.suggestions,
            "feedback_stats": {
                "total_items": len([
                    f for f in self.feedback_items 
                    if f.agent_id == agent_id
                ]),
                "unique_users": len(set(
                    f.user_id for f in self.feedback_items 
                    if f.agent_id == agent_id
                ))
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
    def get_trending_issues(self) -> Dict[str, List[str]]:
        """Identify trending issues across all agents."""
        trends = defaultdict(list)
        
        for feedback in self.feedback_items[-50:]:  # Look at recent feedback
            doc = self.nlp(feedback.text.lower())
            
            # Check for negative sentences
            for sent in doc.sents:
                blob_sent = TextBlob(sent.text)
                if blob_sent.sentiment.polarity < -0.2:
                    trends[feedback.agent_id].append(sent.text.strip())
                    
        return dict(trends)
