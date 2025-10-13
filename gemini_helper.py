"""
Gemini API Integration for Research Analysis
This module uses Google's Gemini API for:
1. Literature review generation
2. Research gap identification
3. Paper summarization
4. Result interpretation
"""

import google.generativeai as genai
import os
from pathlib import Path
import json
from datetime import datetime


class GeminiResearchAssistant:
    """
    AI-powered research assistant using Gemini API
    """
    
    def __init__(self, api_key=None):
        """
        Initialize Gemini API
        
        Args:
            api_key: Gemini API key. If None, reads from environment variable
        """
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Set GEMINI_API_KEY environment variable or pass api_key parameter"
            )
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def generate_literature_review(self, topic, num_papers=10):
        """
        Generate comprehensive literature review for a given topic
        """
        prompt = f"""
        Generate a comprehensive literature review on the following topic:
        "{topic}"
        
        Include:
        1. Introduction and background
        2. Review of {num_papers} recent papers (2020-2025)
        3. Current methodologies and approaches
        4. Key findings and trends
        5. Research gaps and limitations
        6. Future research directions
        
        Format the response in academic style with proper structure.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def identify_research_gaps(self, literature_summary):
        """
        Identify research gaps from literature review
        """
        prompt = f"""
        Based on the following literature review, identify specific research gaps:
        
        {literature_summary}
        
        Provide:
        1. At least 5 specific research gaps
        2. Why each gap is important
        3. Potential impact of addressing each gap
        4. Feasibility of addressing each gap
        
        Format as a numbered list with detailed explanations.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def suggest_novel_algorithm(self, research_gap, existing_methods):
        """
        Suggest novel algorithm or approach
        """
        prompt = f"""
        Research Gap: {research_gap}
        
        Existing Methods: {existing_methods}
        
        Propose a novel algorithm or methodology that addresses this gap.
        Include:
        1. Algorithm name
        2. Core innovation
        3. Step-by-step algorithm description
        4. Expected advantages over existing methods
        5. Potential limitations
        6. Implementation considerations
        
        Be specific and technically detailed.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def generate_research_questions(self, topic, gaps):
        """
        Generate research questions and objectives
        """
        prompt = f"""
        Topic: {topic}
        
        Research Gaps: {gaps}
        
        Generate:
        1. Main research question (1)
        2. Sub-research questions (3-5)
        3. Research objectives (5-7)
        4. Expected contributions
        5. Success criteria
        
        Make them specific, measurable, and achievable.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def analyze_results(self, results_data, methodology):
        """
        Analyze experimental results and provide insights
        """
        prompt = f"""
        Methodology: {methodology}
        
        Results Data: {results_data}
        
        Provide comprehensive analysis:
        1. Key findings and observations
        2. Statistical significance
        3. Comparison with expected outcomes
        4. Strengths of the approach
        5. Limitations and weaknesses
        6. Implications for the field
        7. Recommendations for improvement
        
        Be critical and objective in your analysis.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def suggest_journals(self, paper_topic, paper_abstract):
        """
        Suggest appropriate journals for publication
        """
        prompt = f"""
        Paper Topic: {paper_topic}
        
        Abstract: {paper_abstract}
        
        Suggest 5 appropriate journals for publication:
        - 3 Q2 journals
        - 2 Q3 journals
        
        For each journal provide:
        1. Journal name
        2. Publisher
        3. Quartile (Q2 or Q3)
        4. Impact factor (approximate)
        5. Scope and relevance to the paper
        6. Average publication fee (in USD)
        7. Review timeline
        8. Why this journal is suitable
        
        Focus on cost-effective, Scopus-indexed, SCI journals.
        Prioritize journals commonly cited in medical imaging and deep learning.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def generate_abstract(self, title, methodology, results, conclusion):
        """
        Generate paper abstract
        """
        prompt = f"""
        Generate a 250-word academic abstract for a research paper.
        
        Title: {title}
        Methodology: {methodology}
        Results: {results}
        Conclusion: {conclusion}
        
        Structure:
        - Background and motivation (2-3 sentences)
        - Research gap and objectives (2 sentences)
        - Methodology (3-4 sentences)
        - Key results (3-4 sentences)
        - Conclusion and significance (2 sentences)
        
        Use academic language, be concise and specific.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def improve_paper_section(self, section_name, current_content):
        """
        Improve specific paper section
        """
        prompt = f"""
        Improve the following {section_name} section of a research paper:
        
        Current Content:
        {current_content}
        
        Provide improved version with:
        1. Better academic language
        2. Clearer structure
        3. More precise technical terms
        4. Better flow and transitions
        5. Added relevant details
        
        Maintain factual accuracy.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def generate_comparative_analysis(self, models_results):
        """
        Generate comparative analysis text
        """
        prompt = f"""
        Generate a comprehensive comparative analysis section for a research paper.
        
        Models and Results:
        {models_results}
        
        Include:
        1. Performance comparison across all metrics
        2. Statistical significance of differences
        3. Strengths and weaknesses of each approach
        4. Computational complexity comparison
        5. Practical implications
        6. Recommendations
        
        Use academic writing style with proper citations format.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def save_to_file(self, content, filename):
        """
        Save generated content to file
        """
        output_dir = Path('research_outputs')
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Content saved to: {filepath}")
        return filepath


class ResearchDocumentGenerator:
    """
    Generate complete research document sections
    """
    
    def __init__(self, gemini_assistant):
        self.assistant = gemini_assistant
        
    def generate_complete_paper_outline(self, topic):
        """
        Generate complete paper structure
        """
        sections = {
            'title': '',
            'abstract': '',
            'introduction': '',
            'literature_review': '',
            'methodology': '',
            'results': '',
            'discussion': '',
            'conclusion': '',
            'references': []
        }
        
        print("Generating paper outline...")
        
        # Literature review
        print("1. Generating literature review...")
        sections['literature_review'] = self.assistant.generate_literature_review(topic)
        
        # Research gaps
        print("2. Identifying research gaps...")
        gaps = self.assistant.identify_research_gaps(sections['literature_review'])
        
        # Research questions
        print("3. Generating research questions...")
        sections['research_questions'] = self.assistant.generate_research_questions(
            topic, gaps
        )
        
        print("Paper outline generated!")
        return sections


# Example usage and testing
def main():
    """
    Example usage of Gemini Research Assistant
    """
    print("="*80)
    print("GEMINI RESEARCH ASSISTANT")
    print("="*80)
    
    # Initialize (make sure to set GEMINI_API_KEY environment variable)
    try:
        assistant = GeminiResearchAssistant()
        
        # Example: Generate literature review
        topic = "Hybrid CNN-Transformer architectures for medical image classification"
        
        print("\n1. Generating Literature Review...")
        print("-"*80)
        literature_review = assistant.generate_literature_review(topic, num_papers=10)
        print(literature_review[:500] + "...\n")
        
        # Save to file
        assistant.save_to_file(literature_review, "literature_review.txt")
        
        # Example: Identify research gaps
        print("\n2. Identifying Research Gaps...")
        print("-"*80)
        gaps = assistant.identify_research_gaps(literature_review)
        print(gaps[:500] + "...\n")
        
        assistant.save_to_file(gaps, "research_gaps.txt")
        
        # Example: Suggest journals
        print("\n3. Suggesting Journals for Publication...")
        print("-"*80)
        abstract = """This paper proposes a novel hybrid architecture combining 
        CNNs and Transformers for medical image classification..."""
        
        journals = assistant.suggest_journals(topic, abstract)
        print(journals)
        
        assistant.save_to_file(journals, "suggested_journals.txt")
        
        print("\n" + "="*80)
        print("All outputs saved to 'research_outputs' directory")
        print("="*80)
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTo use Gemini API:")
        print("1. Get API key from: https://makersuite.google.com/app/apikey")
        print("2. Set environment variable:")
        print("   - Linux/Mac: export GEMINI_API_KEY='your-api-key'")
        print("   - Windows: set GEMINI_API_KEY=your-api-key")
        print("   - Or create .env file with: GEMINI_API_KEY=your-api-key")


if __name__ == "__main__":
    main()