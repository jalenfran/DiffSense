"""
Smart Claude Integration - Beautiful AI Analysis
Takes smart context and creates actionable, beautiful responses
"""

import os
import logging
import re
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SmartContext:
    """Context structure for Claude analysis"""
    query: str
    query_type: str
    files: List[Dict[str, Any]]
    commits: List[Dict[str, Any]]
    repo_context: Dict[str, Any]
    confidence: float
    reasoning: str

@dataclass
class SmartResponse:
    """Beautiful response from Smart Claude"""
    query: str
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    context_used: List[str]
    suggestions: List[str]

class ClaudeAnalyzer:
    """Smart Claude that creates beautiful responses from intelligent context"""
    
    def __init__(self):
        self.api_key = os.getenv('CLAUDE_API_KEY')
        # Use a more capable model and much higher token limit for comprehensive analysis
        self.model = os.getenv('CLAUDE_MODEL', 'claude-3-sonnet-20240229')
        self.max_tokens = int(os.getenv('CLAUDE_MAX_TOKENS', '8000'))  # Increased from 3000
        
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.available = True
                logger.info("Smart Claude initialized successfully")
            except ImportError:
                logger.warning("Anthropic package not installed")
                self.available = False
            except Exception as e:
                logger.error(f"Failed to initialize Claude: {e}")
                self.available = False
        else:
            logger.warning("Claude API key not provided")
            self.available = False
    
    def analyze(self, context) -> SmartResponse:
        """Main entry point: receive smart context and produce beautiful analysis"""
        
        if not self.available:
            return self._create_fallback_response(context)
        
        try:
            # Build intelligent prompt
            prompt = self._build_smart_prompt(context)
            
            # Check prompt size and manage if too large
            prompt = self._manage_prompt_size(prompt, context)
            
            # Get Claude's response
            import anthropic
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            claude_text = response.content[0].text
            
            # Create sources from context
            sources = []
            for file_data in context.files:
                sources.append({
                    'type': 'file',
                    'path': file_data['path'],
                    'language': file_data.get('language', 'text'),
                    'size_kb': round(file_data.get('size', 0) / 1024, 1)
                })
            
            for commit in context.commits:
                sources.append({
                    'type': 'commit',
                    'hash': commit.get('hash', '')[:8],
                    'message': commit.get('message', '')[:60] + '...' if len(commit.get('message', '')) > 60 else commit.get('message', ''),
                    'author': commit.get('author', 'Unknown')
                })
            
            # Extract actionable suggestions
            suggestions = self._extract_suggestions(claude_text)
            
            # Create context summary
            context_used = []
            if context.files:
                total_size = sum(f.get('size', 0) for f in context.files) / 1024
                context_used.append(f"{len(context.files)} files ({total_size:.1f}KB)")
            
            if context.commits:
                context_used.append(f"{len(context.commits)} commits")
            
            return SmartResponse(
                query=context.query,
                response=claude_text,
                confidence=min(context.confidence + 0.1, 0.95),
                sources=sources,
                context_used=context_used,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            return self._create_fallback_response(context)
    
    def _build_smart_prompt(self, context) -> str:
        """Build intelligent prompt with perfect context"""
        
        prompt_parts = [
            f"# 🔍 Smart Code Analysis",
            f"",
            f"**Query**: {context.query}",
            f"**Analysis Type**: {context.query_type}",
            f"**Context Quality**: {context.confidence:.0%}",
            f"**Reasoning**: {context.reasoning}",
            f"",
        ]
        
        # Add repository context
        repo = context.repo_context
        prompt_parts.extend([
            f"## 📊 Repository Overview",
            f"- **ID**: {repo.get('repo_id', 'Unknown')}",
            f"- **Total Files**: {repo.get('total_files', 0)}",
            f"- **Total Commits**: {repo.get('total_commits', 0)}",
            f"- **Recent Activity**: {repo.get('recent_activity', 0)} commits (last 30 days)",
            f"- **Main Languages**: {', '.join(list(repo.get('languages', {}).keys())[:3])}",
            f"",
        ])
        
        # Add file context
        if context.files:
            prompt_parts.extend([
                f"## 📁 Files Analyzed ({len(context.files)} files)",
                f"",
            ])
            
            # Include ALL files, not just top 3
            for i, file_data in enumerate(context.files):
                size_kb = file_data.get('size', 0) / 1024
                content = file_data['content']
                
                # Only truncate extremely large files (>50KB)
                if len(content) > 50000:
                    content = content[:50000] + f"\n\n... [File truncated - total size: {size_kb:.1f}KB] ..."
                
                prompt_parts.extend([
                    f"### {i+1}. `{file_data['path']}` ({file_data.get('language', 'text')}, {size_kb:.1f}KB)",
                    f"```{file_data.get('language', 'text')}",
                    content,
                    f"```",
                    f"",
                ])
        
        # Add commit context
        if context.commits:
            prompt_parts.extend([
                f"## 📝 Commits Analyzed ({len(context.commits)} commits)",
                f"",
            ])
            
            # Include ALL commits, not just top 5
            for i, commit in enumerate(context.commits):
                prompt_parts.extend([
                    f"### {i+1}. Commit `{commit.get('hash', '')[:8]}` by {commit.get('author', 'Unknown')}",
                    f"**Date**: {commit.get('date', 'Unknown')[:10]}",
                    f"**Message**: {commit.get('message', 'No message')}",
                    f"**Files Changed**: {len(commit.get('files_changed', []))}",
                    f"",
                ])
                
                # Show ALL file changes/diffs, not just top 2
                for file_change in commit.get('files_changed', []):
                    diff_content = file_change.get('diff', '')
                    
                    # Only truncate extremely large diffs (>20KB)
                    if len(diff_content) > 20000:
                        diff_content = diff_content[:20000] + "\n... [diff truncated] ..."
                    
                    if diff_content.strip():
                        prompt_parts.extend([
                            f"#### Changes to `{file_change.get('file', 'unknown')}`",
                            f"```diff",
                            diff_content,
                            f"```",
                            f"",
                        ])
        
        # Add analysis instructions
        prompt_parts.extend([
            f"## 🎯 Analysis Instructions",
            f"",
            f"Please provide a **concise, actionable response** that directly answers: '{context.query}'",
            f"",
            f"**Format your response as follows:**",
            f"",
            f"### 🔍 Direct Answer",
            f"[Directly answer the specific question asked]",
            f"",
            f"### 🔑 Key Findings",
            f"- [Most important discovery 1]",
            f"- [Most important discovery 2]",
            f"- [Most important discovery 3]",
            f"",
            f"### ⚠️ Issues Found",
            f"- [Specific issue with file:line reference if applicable]",
            f"- [Another issue if found]",
            f"",
            f"### 💡 Recommendations",
            f"1. [Concrete actionable step 1]",
            f"2. [Concrete actionable step 2]",
            f"3. [Concrete actionable step 3]",
            f"",
            f"### 🎯 Priority",
            f"[High/Medium/Low] - [Brief reasoning]",
            f"",
            f"**Keep it practical, developer-friendly, and under 400 words.**"
        ])

        additional_instructions = [
            "Focus on:",
            "1. **Direct Answer** - Address the specific question asked",
            "2. **Key Findings** - Most important discoveries (2-3 bullet points)",
            "3. **Specific Issues** - Any problems found with file/line references",
            "4. **Actionable Recommendations** - Concrete next steps",
            "5. **Risk Assessment** - Priority level and reasoning",
            "",
            "Keep the response practical, developer-friendly, and under 500 words."
        ]
        
        return "\n".join(prompt_parts + additional_instructions)
    
    def _extract_suggestions(self, claude_text: str) -> List[str]:
        """Extract actionable suggestions from Claude's response"""
        suggestions = []
        
        # Look for recommendations section
        lines = claude_text.split('\n')
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            
            # Check if we're in recommendations section
            if 'recommendation' in line.lower() or 'action' in line.lower():
                in_recommendations = True
                continue
            
            # Extract numbered/bulleted suggestions
            if in_recommendations and line:
                if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']):
                    # Clean up the suggestion
                    clean_suggestion = line
                    for prefix in ['1. ', '2. ', '3. ', '4. ', '5. ', '- ', '* ', '• ']:
                        if clean_suggestion.startswith(prefix):
                            clean_suggestion = clean_suggestion[len(prefix):].strip()
                            break
                    
                    if len(clean_suggestion) > 10 and len(clean_suggestion) < 200:
                        suggestions.append(clean_suggestion)
            
            # Stop if we hit another section
            if line.startswith('#') and in_recommendations:
                break
        
        # If no recommendations section found, look for general action words
        if not suggestions:
            for line in lines:
                line = line.strip()
                if any(action in line.lower() for action in ['should', 'review', 'check', 'update', 'fix', 'refactor', 'consider']):
                    if len(line) > 20 and len(line) < 200:
                        suggestions.append(line)
        
        return suggestions[:4]  # Return top 4 suggestions
    
    def _create_fallback_response(self, context) -> SmartResponse:
        """Create fallback response when Claude is not available"""
        
        response_parts = [
            f"# 📊 Smart Analysis Results",
            f"",
            f"**Query**: {context.query}",
            f"**Analysis Type**: {context.query_type}",
            f"**Context Quality**: {context.confidence:.0%}",
            f"",
        ]
        
        # Show what we found
        if context.files:
            response_parts.extend([
                f"## 📁 Files Found ({len(context.files)})",
                f""
            ])
            for file_data in context.files[:3]:
                size_kb = file_data.get('size', 0) / 1024
                response_parts.append(f"- **{file_data['path']}** ({file_data.get('language', 'text')}, {size_kb:.1f}KB)")
            response_parts.append("")
        
        if context.commits:
            response_parts.extend([
                f"## 📝 Commits Found ({len(context.commits)})",
                f""
            ])
            for commit in context.commits[:3]:
                response_parts.append(f"- **{commit.get('hash', '')[:8]}**: {commit.get('message', '')[:60]}...")
            response_parts.append("")
        
        response_parts.extend([
            f"## 💡 Next Steps",
            f"",
            f"1. **Configure Claude API** for enhanced AI analysis",
            f"2. **Review the gathered files** manually for insights",
            f"3. **Examine the commit history** for relevant changes",
            f"",
            f"*Smart context gathering completed successfully - Claude API needed for AI analysis.*"
        ])
        
        # Create sources
        sources = []
        for file_data in context.files:
            sources.append({
                'type': 'file',
                'path': file_data['path'],
                'language': file_data.get('language', 'text'),
                'size_kb': round(file_data.get('size', 0) / 1024, 1)
            })
        
        for commit in context.commits:
            sources.append({
                'type': 'commit',
                'hash': commit.get('hash', '')[:8],
                'message': commit.get('message', '')[:50],
                'author': commit.get('author', 'Unknown')
            })
        
        # Create basic suggestions
        suggestions = ["Configure Claude API key for AI-powered analysis"]
        if context.files:
            suggestions.append(f"Review the {len(context.files)} relevant files manually")
        if context.commits:
            suggestions.append(f"Examine the {len(context.commits)} related commits")
        
        return SmartResponse(
            query=context.query,
            response="\n".join(response_parts),
            confidence=context.confidence * 0.7,  # Lower without Claude
            sources=sources,
            context_used=[f"Smart context gathered: {context.reasoning}"],
            suggestions=suggestions
        )
    
    def enhance_rag_response(self, query: str, rag_result, repository_context: Dict[str, Any]) -> str:
        """Enhance RAG response with Claude insights"""
        if not self.available:
            return rag_result.response + "\n\n💡 Configure Claude API key for AI-enhanced analysis."
        
        try:
            import anthropic
            
            prompt = f"""Enhance this repository analysis response:

USER QUERY: {query}

CURRENT RESPONSE: {rag_result.response}

REPOSITORY: {repository_context.get('repo_id', 'Unknown')}

Please provide additional insights, technical context, and actionable recommendations.
Keep the response concise (2-3 paragraphs) and practical."""

            print({"role": "user", "content": prompt})
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            
            enhanced_text = response.content[0].text
            return f"{rag_result.response}\n\n🤖 **AI Enhancement:**\n{enhanced_text}"
            
        except Exception as e:
            logger.error(f"RAG enhancement failed: {e}")
            return rag_result.response + f"\n\n⚠️ AI enhancement failed: {str(e)}"
    
    def analyze_commit_breaking_changes(self, commit_analysis, diff_content: str = "") -> Dict[str, Any]:
        """Analyze commit for breaking changes with Claude"""
        if not self.available:
            return {
                "claude_analysis": {"status": "unavailable", "summary": "Claude API not configured"},
                "confidence": 0.0,
                "recommendations": ["Configure Claude API for enhanced analysis"]
            }
        
        try:
            import anthropic
            
            prompt = f"""Analyze this commit for breaking changes:

COMMIT: {commit_analysis.commit_hash}
MESSAGE: {commit_analysis.commit_message}
AUTHOR: {commit_analysis.author}
RISK SCORE: {commit_analysis.overall_risk_score}

{diff_content[:1000] if diff_content else "No diff content provided"}

Provide JSON response with risk_level, summary, concerns, and recommendations."""
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis_text = response.content[0].text
            
            return {
                "claude_analysis": {"summary": analysis_text},
                "confidence": 0.8,
                "recommendations": ["Review the analysis for actionable insights"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Commit analysis failed: {e}")
            return {
                "claude_analysis": {"status": "error", "error": str(e)},
                "confidence": 0.0,
                "recommendations": ["Retry analysis or check logs"]
            }
    
    def analyze_repository_risk(self, repo_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze repository risk with Claude"""
        if not self.available:
            return {
                "risk_assessment": {"status": "unavailable"},
                "overall_risk_level": "unknown",
                "recommendations": ["Configure Claude API for risk analysis"]
            }
        
        try:
            import anthropic
            
            stats = repo_summary.get("statistics", {})
            prompt = f"""Analyze repository risk:

TOTAL COMMITS: {stats.get('total_commits_indexed', 0)}
TOTAL FILES: {stats.get('total_files_indexed', 0)}
HIGH RISK COMMITS: {stats.get('high_risk_commits', 0)}

Provide JSON with risk_level (low/medium/high), summary, concerns, and recommendations."""
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis_text = response.content[0].text
            
            return {
                "risk_assessment": {"summary": analysis_text},
                "overall_risk_level": "medium",
                "recommendations": ["Review analysis for specific concerns"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Repository risk analysis failed: {e}")
            return {
                "risk_assessment": {"status": "error", "error": str(e)},
                "overall_risk_level": "unknown",
                "recommendations": ["Retry analysis or check logs"]
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get Claude analyzer status"""
        return {
            "available": self.available,
            "model": self.model if self.available else None,
            "max_tokens": self.max_tokens,
            "api_key_configured": bool(self.api_key)
        }
    
    def _manage_prompt_size(self, prompt: str, context) -> str:
        """Intelligently manage prompt size for Claude's limits"""
        
        # Claude 3 Sonnet has ~200K token context window
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = 600000  # Conservative limit (~150K tokens)
        
        # Log what we're feeding to Claude for debugging
        logger.info(f"🔍 CLAUDE DEBUG - Preparing prompt for analysis:")
        logger.info(f"  📊 Query: {context.query}")
        logger.info(f"  📁 Files to analyze: {len(context.files)}")
        logger.info(f"  📝 Commits to analyze: {len(context.commits)}")
        
        if context.files:
            total_file_size = sum(len(f.get('content', '')) for f in context.files)
            logger.info(f"  📏 Total file content size: {total_file_size:,} characters")
            for i, file_data in enumerate(context.files):
                file_size = len(file_data.get('content', ''))
                logger.info(f"    {i+1}. {file_data['path']} - {file_size:,} chars ({file_data.get('language', 'text')})")
        
        if context.commits:
            total_diff_size = sum(len(str(c.get('files_changed', []))) for c in context.commits)
            logger.info(f"  📏 Total commit diff size: {total_diff_size:,} characters")
            for i, commit in enumerate(context.commits):
                diff_size = sum(len(fc.get('diff', '')) for fc in commit.get('files_changed', []))
                logger.info(f"    {i+1}. {commit.get('hash', '')[:8]} - {diff_size:,} chars diff")
        
        current_size = len(prompt)
        logger.info(f"  📐 Total prompt size: {current_size:,} characters")
        
        if current_size <= max_chars:
            logger.info(f"  ✅ Prompt fits within Claude limits, sending all content")
            return prompt
        
        logger.warning(f"  ⚠️ Prompt too large ({current_size:,} chars), need to trim")
        
        # If prompt is too large, we need to intelligently trim
        # Priority: Keep query context, trim file content first, then commits
        
        # Try reducing file content first
        if context.files:
            logger.info(f"  🔧 Trimming file content to fit Claude limits")
            total_files_budget = max_chars // 2  # Use half budget for files
            chars_per_file = total_files_budget // len(context.files)
            
            for file_data in context.files:
                original_size = len(file_data['content'])
                if original_size > chars_per_file:
                    # Keep beginning and end of file
                    keep_size = chars_per_file // 2
                    file_data['content'] = (
                        file_data['content'][:keep_size] + 
                        f"\n\n... [FILE TRUNCATED - Original size: {original_size:,} chars] ...\n\n" +
                        file_data['content'][-keep_size:]
                    )
                    logger.info(f"    📝 Trimmed {file_data['path']} from {original_size:,} to {len(file_data['content']):,} chars")
        
        # Rebuild prompt with trimmed content
        trimmed_prompt = self._build_smart_prompt(context)
        final_size = len(trimmed_prompt)
        logger.info(f"  📐 Final prompt size after trimming: {final_size:,} characters")
        
        return trimmed_prompt

    # ...existing code...
