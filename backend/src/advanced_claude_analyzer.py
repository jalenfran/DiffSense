"""
Advanced Claude Analyzer - Domain Expert AI with Intelligent Prompting
Provides expert-level analysis with specialized knowledge and smart reasoning
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ExpertAnalysis:
    """Expert-level analysis with domain specialization"""
    domain: str
    expert_level: str  # 'senior', 'principal', 'architect'
    analysis: str
    confidence: float
    risk_assessment: Dict[str, Any]
    actionable_recommendations: List[Dict[str, Any]]
    code_examples: List[Dict[str, str]]
    metrics: Dict[str, float]
    follow_up_questions: List[str]

class AdvancedClaudeAnalyzer:
    """Advanced Claude analyzer with domain expertise and intelligent prompting"""
    
    def __init__(self):
        self.api_key = os.getenv('CLAUDE_API_KEY')
        self.model = os.getenv('CLAUDE_MODEL', 'claude-3-sonnet-20240229')
        self.max_tokens = int(os.getenv('CLAUDE_MAX_TOKENS', '8000'))
        
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.available = True
                logger.info("Advanced Claude analyzer initialized successfully")
            except ImportError:
                logger.warning("Anthropic package not installed")
                self.available = False
            except Exception as e:
                logger.error(f"Failed to initialize Claude: {e}")
                self.available = False
        else:
            logger.warning("Claude API key not provided")
            self.available = False
        
        # Expert personas for different domains
        self.expert_personas = {
            'security': {
                'title': 'Senior Security Engineer',
                'years_experience': '10+ years',
                'specialties': ['Application Security', 'Vulnerability Assessment', 'Secure Code Review', 'OWASP'],
                'certifications': ['CISSP', 'CEH', 'OSCP'],
                'thinking_style': 'threat-modeling, risk-assessment, defense-in-depth'
            },
            'architecture': {
                'title': 'Principal Software Architect',
                'years_experience': '15+ years',
                'specialties': ['System Design', 'Microservices', 'Scalability', 'Design Patterns'],
                'certifications': ['AWS Solutions Architect', 'Azure Architect'],
                'thinking_style': 'systems-thinking, scalability-first, pattern-recognition'
            },
            'performance': {
                'title': 'Senior Performance Engineer',
                'years_experience': '12+ years',
                'specialties': ['Performance Optimization', 'Profiling', 'Scalability', 'Distributed Systems'],
                'certifications': ['Performance Engineering', 'Site Reliability Engineering'],
                'thinking_style': 'metrics-driven, bottleneck-identification, optimization-focused'
            },
            'quality': {
                'title': 'Staff Software Engineer (Code Quality)',
                'years_experience': '8+ years',
                'specialties': ['Code Review', 'Technical Debt', 'Refactoring', 'Testing Strategies'],
                'certifications': ['Clean Code', 'Software Craftsmanship'],
                'thinking_style': 'maintainability-focused, technical-debt-aware, quality-metrics'
            },
            'devops': {
                'title': 'DevOps Architect',
                'years_experience': '10+ years',
                'specialties': ['CI/CD', 'Infrastructure as Code', 'Monitoring', 'Automation'],
                'certifications': ['AWS DevOps', 'Kubernetes', 'Docker'],
                'thinking_style': 'automation-first, reliability-focused, infrastructure-as-code'
            }
        }
        
        # Advanced prompting techniques
        self.prompting_techniques = {
            'chain_of_thought': True,
            'step_by_step_reasoning': True,
            'few_shot_examples': True,
            'structured_output': True,
            'domain_expertise': True
        }
    
    def expert_analysis(self, query: str, domain: str, context: Dict[str, Any], 
                       file_analyses: List[Any], specialized_data: Dict[str, Any]) -> ExpertAnalysis:
        """Perform expert-level analysis with domain specialization"""
        
        if not self.available:
            return self._create_fallback_expert_analysis(query, domain, context)
        
        try:
            # Create expert persona and prompt
            expert_prompt = self._create_expert_persona_prompt(domain, query, context, file_analyses, specialized_data)
            
            # Execute analysis with Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": expert_prompt}]
            )
            
            analysis_text = response.content[0].text
            
            # Parse structured response
            parsed_analysis = self._parse_expert_response(analysis_text, domain)
            
            return ExpertAnalysis(
                domain=domain,
                expert_level=self.expert_personas[domain]['title'],
                analysis=parsed_analysis['analysis'],
                confidence=parsed_analysis['confidence'],
                risk_assessment=parsed_analysis['risk_assessment'],
                actionable_recommendations=parsed_analysis['recommendations'],
                code_examples=parsed_analysis['code_examples'],
                metrics=parsed_analysis['metrics'],
                follow_up_questions=parsed_analysis['follow_up_questions']
            )
            
        except Exception as e:
            logger.error(f"Expert analysis failed: {e}")
            return self._create_fallback_expert_analysis(query, domain, context)
    
    def _create_expert_persona_prompt(self, domain: str, query: str, context: Dict[str, Any], 
                                    file_analyses: List[Any], specialized_data: Dict[str, Any]) -> str:
        """Create expert persona prompt with advanced techniques"""
        
        persona = self.expert_personas.get(domain, self.expert_personas['quality'])
        
        # Build the expert persona introduction
        persona_intro = f"""# 🎯 Expert Analysis Session

## Your Expert Identity
You are a **{persona['title']}** with {persona['years_experience']} of industry experience.

**Your Specialties:**
{chr(10).join(f"- {specialty}" for specialty in persona['specialties'])}

**Certifications:**
{chr(10).join(f"- {cert}" for cert in persona['certifications'])}

**Thinking Style:** {persona['thinking_style']}

**Analysis Approach:** Provide expert-level insights that only a senior professional with your background would offer. Be specific, actionable, and demonstrate deep technical understanding.

---"""

        # Add query context
        query_context = f"""
## 📋 Analysis Request
**Query:** {query}
**Domain Focus:** {domain.title()}
**Context Quality:** {context.get('confidence', 0.5):.0%}

"""

        # Add specialized analysis data
        specialized_context = ""
        if domain in specialized_data:
            data = specialized_data[domain]
            specialized_context = f"""
## 🔍 {domain.title()} Analysis Data
{self._format_specialized_data(domain, data)}

"""

        # Add file analysis context
        files_context = ""
        if file_analyses:
            files_context = f"""
## 📁 Code Analysis Results
**Files Analyzed:** {len(file_analyses)}

"""
            
            for i, analysis in enumerate(file_analyses[:5]):  # Top 5 files
                files_context += f"""
### File {i+1}: `{analysis.file_path}`
- **Language:** {analysis.language}
- **Complexity Score:** {analysis.complexity_score:.1f}/10
- **Entities:** {len(analysis.entities)} classes/functions
- **Security Patterns:** {len(analysis.security_patterns)} issues detected
- **Architecture Patterns:** {', '.join(analysis.architecture_patterns) or 'None detected'}
- **Technical Debt:** {', '.join(analysis.technical_debt_indicators[:3]) or 'None detected'}

"""

        # Add domain-specific prompting
        domain_specific_prompt = self._get_domain_specific_prompt(domain, specialized_data.get(domain, {}))
        
        # Chain of thought reasoning prompt
        reasoning_prompt = f"""
## 🧠 Expert Analysis Framework

**Step 1: Initial Assessment**
- What are the most critical aspects to analyze based on my expertise?
- What patterns or anti-patterns do I immediately recognize?

**Step 2: Deep Technical Analysis**
- What specific technical issues require immediate attention?
- How do these issues align with industry best practices in my domain?

**Step 3: Risk Assessment**
- What are the highest priority risks from my expert perspective?
- What could happen if these issues are not addressed?

**Step 4: Expert Recommendations**
- What specific, actionable steps would I recommend?
- What tools, practices, or architectural changes are needed?

**Step 5: Implementation Strategy**
- How should these recommendations be prioritized?
- What's the most effective implementation approach?

---

"""

        # Output format specification
        output_format = f"""
## 📝 Required Output Format

Please structure your expert analysis exactly as follows:

### 🔍 EXPERT ASSESSMENT
[Provide your senior-level assessment of the current state]

### 🚨 CRITICAL ISSUES
[List the most important issues that need immediate attention, with priority levels]

### 📊 METRICS & MEASUREMENTS
[Provide specific metrics and measurements relevant to {domain}]

### 💡 EXPERT RECOMMENDATIONS
[Provide 3-5 specific, actionable recommendations with implementation details]

### 🛠️ CODE EXAMPLES
[If applicable, provide specific code examples or patterns]

### ⚠️ RISK ASSESSMENT
[Assess risks with High/Medium/Low priorities and business impact]

### 🎯 NEXT STEPS
[Provide a prioritized action plan]

### ❓ FOLLOW-UP QUESTIONS
[Suggest 2-3 follow-up questions for deeper analysis]

**Remember:** Your analysis should demonstrate the deep expertise of a {persona['title']} with {persona['years_experience']} experience. Be specific, technical, and actionable.

"""

        # Combine all parts
        full_prompt = (
            persona_intro + 
            query_context + 
            specialized_context + 
            files_context + 
            domain_specific_prompt + 
            reasoning_prompt + 
            output_format
        )
        
        return full_prompt
    
    def _format_specialized_data(self, domain: str, data: Dict[str, Any]) -> str:
        """Format specialized data for the prompt"""
        formatted = ""
        
        if domain == 'security':
            formatted = f"""
**Risk Level:** {data.get('risk_level', 'unknown').upper()}
**Security Issues Found:** {len(data.get('issues_found', []))}
**Affected Files:** {len(data.get('affected_files', []))}

**Critical Security Patterns:**
{chr(10).join(f"- {issue}" for issue in data.get('issues_found', [])[:5])}

**Recent Security Commits:** {len(data.get('recent_security_commits', []))}
"""
        
        elif domain == 'architecture':
            formatted = f"""
**Patterns Detected:** {', '.join(data.get('patterns_detected', []))}
**Entry Points:** {len(data.get('entry_points', []))}
**Potential Bottlenecks:** {len(data.get('potential_bottlenecks', []))}

**Complexity Distribution:**
{chr(10).join(f"- {file}: {complexity:.1f}" for file, complexity in data.get('complexity_distribution', [])[:5])}

**Layer Structure:**
{chr(10).join(f"- {pattern}: {len(files)} files" for pattern, files in data.get('layer_structure', {}).items())}
"""
        
        elif domain == 'performance':
            formatted = f"""
**Potential Bottlenecks:** {len(data.get('potential_bottlenecks', []))}
**Files Needing Optimization:** {len(data.get('files_needing_optimization', []))}

**Performance Hotspots:**
{chr(10).join(f"- {file}: Complexity {complexity:.1f}" for file, complexity in data.get('potential_bottlenecks', [])[:5])}
"""
        
        elif domain == 'quality':
            formatted = f"""
**Quality Score:** {data.get('quality_score', 0):.1f}/10
**High Complexity Files:** {len(data.get('high_complexity_files', []))}
**Technical Debt Files:** {len(data.get('technical_debt_files', []))}
**Total LOC:** {data.get('total_lines_of_code', 0):,}

**Quality Issues:**
{chr(10).join(f"- {file}: Complexity {complexity:.1f}" for file, complexity in data.get('high_complexity_files', [])[:5])}
"""
        
        return formatted
    
    def _get_domain_specific_prompt(self, domain: str, data: Dict[str, Any]) -> str:
        """Get domain-specific expert prompting"""
        
        domain_prompts = {
            'security': f"""
## 🛡️ Security Expert Context

As a security professional, focus on:

**Threat Modeling:** Consider the STRIDE model (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege)

**Security Frameworks:** Apply OWASP Top 10, SANS, and NIST guidelines

**Key Security Questions:**
- What are the attack vectors in this code?
- How could an attacker exploit these vulnerabilities?
- What security controls are missing?
- How would you prioritize these security fixes?

**Think like an attacker:** What would you target first? How would you exploit these weaknesses?

""",
            
            'architecture': f"""
## 🏗️ Architecture Expert Context

As a principal architect, focus on:

**Design Principles:** SOLID, DRY, KISS, YAGNI
**Architecture Patterns:** Layered, Microservices, Event-Driven, CQRS
**Quality Attributes:** Scalability, Maintainability, Reliability, Security

**Key Architecture Questions:**
- How well does this architecture support business requirements?
- What are the scalability limitations?
- How maintainable is this design?
- What technical debt is accumulating?

**Think strategically:** How will this system evolve? What are the long-term implications?

""",
            
            'performance': f"""
## ⚡ Performance Expert Context

As a performance engineer, focus on:

**Performance Principles:** Measure first, optimize bottlenecks, profile continuously
**Key Metrics:** Latency, Throughput, Resource Utilization, Scalability
**Optimization Techniques:** Caching, Async processing, Database optimization, CDN

**Key Performance Questions:**
- Where are the performance bottlenecks?
- How will this perform under load?
- What are the resource consumption patterns?
- How can we improve efficiency?

**Think data-driven:** What metrics matter most? How can we measure and improve?

""",
            
            'quality': f"""
## 📊 Code Quality Expert Context

As a code quality expert, focus on:

**Quality Metrics:** Complexity, Maintainability, Test Coverage, Technical Debt
**Best Practices:** Clean Code, Refactoring, Code Reviews, Documentation
**Quality Gates:** Automated checks, Standards compliance, Review processes

**Key Quality Questions:**
- How maintainable is this code?
- What technical debt exists?
- How can readability be improved?
- What refactoring opportunities exist?

**Think long-term:** How will this code age? What will future developers think?

"""
        }
        
        return domain_prompts.get(domain, "")
    
    def _parse_expert_response(self, response_text: str, domain: str) -> Dict[str, Any]:
        """Parse structured expert response"""
        
        parsed = {
            'analysis': response_text,
            'confidence': 0.8,
            'risk_assessment': {},
            'recommendations': [],
            'code_examples': [],
            'metrics': {},
            'follow_up_questions': []
        }
        
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Detect sections
            if '### 🔍 EXPERT ASSESSMENT' in line or 'EXPERT ASSESSMENT' in line:
                current_section = 'assessment'
            elif '### 🚨 CRITICAL ISSUES' in line or 'CRITICAL ISSUES' in line:
                current_section = 'issues'
            elif '### 📊 METRICS' in line or 'METRICS' in line:
                current_section = 'metrics'
            elif '### 💡 EXPERT RECOMMENDATIONS' in line or 'RECOMMENDATIONS' in line:
                current_section = 'recommendations'
            elif '### ⚠️ RISK ASSESSMENT' in line or 'RISK ASSESSMENT' in line:
                current_section = 'risk'
            elif '### ❓ FOLLOW-UP' in line or 'FOLLOW-UP' in line:
                current_section = 'follow_up'
            elif line.startswith('###'):
                current_section = None
            
            # Parse content based on section
            if current_section == 'recommendations' and line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*')):
                recommendation = self._extract_recommendation(line)
                if recommendation:
                    parsed['recommendations'].append(recommendation)
            
            elif current_section == 'risk' and any(priority in line.upper() for priority in ['HIGH', 'MEDIUM', 'LOW']):
                risk_item = self._extract_risk_item(line)
                if risk_item:
                    parsed['risk_assessment'][risk_item['item']] = risk_item
            
            elif current_section == 'follow_up' and line.startswith(('1.', '2.', '3.', '-', '*')):
                question = self._extract_follow_up_question(line)
                if question:
                    parsed['follow_up_questions'].append(question)
        
        # Extract code examples
        parsed['code_examples'] = self._extract_code_examples(response_text)
        
        # Calculate confidence based on response quality
        parsed['confidence'] = self._calculate_response_confidence(response_text, domain)
        
        return parsed
    
    def _extract_recommendation(self, line: str) -> Optional[Dict[str, Any]]:
        """Extract structured recommendation from line"""
        # Remove list markers
        clean_line = line
        for marker in ['1. ', '2. ', '3. ', '4. ', '5. ', '- ', '* ', '• ']:
            if clean_line.startswith(marker):
                clean_line = clean_line[len(marker):].strip()
                break
        
        if len(clean_line) < 10:
            return None
        
        # Determine priority
        priority = 'medium'
        if any(word in clean_line.lower() for word in ['critical', 'urgent', 'immediate', 'asap']):
            priority = 'high'
        elif any(word in clean_line.lower() for word in ['consider', 'optional', 'future', 'nice']):
            priority = 'low'
        
        # Determine category
        category = 'general'
        if any(word in clean_line.lower() for word in ['security', 'vulnerability', 'auth']):
            category = 'security'
        elif any(word in clean_line.lower() for word in ['performance', 'optimize', 'speed']):
            category = 'performance'
        elif any(word in clean_line.lower() for word in ['refactor', 'quality', 'clean']):
            category = 'quality'
        
        return {
            'text': clean_line,
            'priority': priority,
            'category': category,
            'actionable': True
        }
    
    def _extract_risk_item(self, line: str) -> Optional[Dict[str, str]]:
        """Extract risk assessment item"""
        line_upper = line.upper()
        
        if 'HIGH' in line_upper:
            priority = 'high'
        elif 'MEDIUM' in line_upper:
            priority = 'medium'
        elif 'LOW' in line_upper:
            priority = 'low'
        else:
            return None
        
        # Extract the risk description
        item_text = line.strip()
        for marker in ['- ', '* ', '• ']:
            if item_text.startswith(marker):
                item_text = item_text[len(marker):].strip()
                break
        
        return {
            'item': item_text,
            'priority': priority,
            'description': item_text
        }
    
    def _extract_follow_up_question(self, line: str) -> Optional[str]:
        """Extract follow-up question"""
        clean_line = line
        for marker in ['1. ', '2. ', '3. ', '- ', '* ', '• ']:
            if clean_line.startswith(marker):
                clean_line = clean_line[len(marker):].strip()
                break
        
        if len(clean_line) > 10 and ('?' in clean_line or 'how' in clean_line.lower() or 'what' in clean_line.lower()):
            return clean_line
        
        return None
    
    def _extract_code_examples(self, text: str) -> List[Dict[str, str]]:
        """Extract code examples from response"""
        examples = []
        
        # Find code blocks
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        
        for language, code in matches:
            if code.strip():
                examples.append({
                    'language': language or 'text',
                    'code': code.strip(),
                    'context': 'expert_recommendation'
                })
        
        return examples
    
    def _calculate_response_confidence(self, response_text: str, domain: str) -> float:
        """Calculate confidence based on response quality"""
        confidence = 0.5  # Base confidence
        
        # Check for structured sections
        if '### 🔍 EXPERT ASSESSMENT' in response_text or 'EXPERT ASSESSMENT' in response_text:
            confidence += 0.1
        if '### 💡 EXPERT RECOMMENDATIONS' in response_text or 'RECOMMENDATIONS' in response_text:
            confidence += 0.1
        if '### ⚠️ RISK ASSESSMENT' in response_text or 'RISK ASSESSMENT' in response_text:
            confidence += 0.1
        
        # Check for domain-specific expertise
        domain_indicators = {
            'security': ['vulnerability', 'attack', 'threat', 'owasp', 'security'],
            'architecture': ['pattern', 'scalability', 'design', 'architecture', 'solid'],
            'performance': ['bottleneck', 'optimization', 'latency', 'throughput', 'performance'],
            'quality': ['refactor', 'maintainability', 'technical debt', 'complexity', 'clean code']
        }
        
        indicators = domain_indicators.get(domain, [])
        indicator_count = sum(1 for indicator in indicators if indicator in response_text.lower())
        confidence += min(indicator_count * 0.05, 0.2)
        
        # Check response length and detail
        if len(response_text) > 1000:
            confidence += 0.05
        if len(response_text) > 2000:
            confidence += 0.05
        
        return min(confidence, 0.95)
    
    def _create_fallback_expert_analysis(self, query: str, domain: str, context: Dict[str, Any]) -> ExpertAnalysis:
        """Create fallback analysis when Claude is not available"""
        
        persona = self.expert_personas.get(domain, self.expert_personas['quality'])
        
        fallback_analysis = f"""# Expert Analysis (Fallback Mode)

As a {persona['title']}, I would analyze this query: "{query}"

Unfortunately, Claude API is not available for detailed expert analysis.

## Recommendations:
1. Configure Claude API key for expert-level analysis
2. Review the code manually using {domain} best practices
3. Consider using automated {domain} analysis tools

**Expert Context:** {persona['specialties']}
"""
        
        return ExpertAnalysis(
            domain=domain,
            expert_level=persona['title'],
            analysis=fallback_analysis,
            confidence=0.3,
            risk_assessment={'api_unavailable': {'priority': 'medium', 'description': 'Claude API not configured'}},
            actionable_recommendations=[
                {'text': 'Configure Claude API key for enhanced analysis', 'priority': 'high', 'category': 'setup'}
            ],
            code_examples=[],
            metrics={'confidence': 0.3},
            follow_up_questions=[
                'What specific aspects of the code need expert review?',
                f'What {domain}-specific tools could provide automated analysis?'
            ]
        )
    
    def multi_expert_consultation(self, query: str, context: Dict[str, Any], 
                                 file_analyses: List[Any], domains: List[str]) -> Dict[str, ExpertAnalysis]:
        """Consult multiple experts for comprehensive analysis"""
        
        expert_analyses = {}
        
        for domain in domains:
            specialized_data = context.get('specialized_analysis', {})
            expert_analysis = self.expert_analysis(
                query, domain, context, file_analyses, specialized_data
            )
            expert_analyses[domain] = expert_analysis
        
        return expert_analyses
    
    def synthesize_expert_opinions(self, expert_analyses: Dict[str, ExpertAnalysis]) -> Dict[str, Any]:
        """Synthesize multiple expert opinions into unified insights"""
        
        synthesis = {
            'consensus_recommendations': [],
            'conflicting_opinions': [],
            'priority_matrix': {},
            'unified_risk_assessment': {},
            'cross_domain_insights': []
        }
        
        # Find consensus recommendations
        all_recommendations = []
        for domain, analysis in expert_analyses.items():
            for rec in analysis.actionable_recommendations:
                rec['source_domain'] = domain
                all_recommendations.append(rec)
        
        # Group similar recommendations
        recommendation_groups = self._group_similar_recommendations(all_recommendations)
        synthesis['consensus_recommendations'] = recommendation_groups
        
        # Combine risk assessments
        for domain, analysis in expert_analyses.items():
            for risk_item, risk_data in analysis.risk_assessment.items():
                if risk_item in synthesis['unified_risk_assessment']:
                    # Merge risk assessments
                    existing = synthesis['unified_risk_assessment'][risk_item]
                    if risk_data['priority'] == 'high' or existing['priority'] == 'high':
                        existing['priority'] = 'high'
                    existing['domains'].append(domain)
                else:
                    risk_data['domains'] = [domain]
                    synthesis['unified_risk_assessment'][risk_item] = risk_data
        
        return synthesis
    
    def _group_similar_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group similar recommendations from different experts"""
        # Simple similarity grouping based on keywords
        grouped = []
        processed = set()
        
        for i, rec1 in enumerate(recommendations):
            if i in processed:
                continue
            
            group = {
                'primary_recommendation': rec1['text'],
                'priority': rec1['priority'],
                'supporting_domains': [rec1['source_domain']],
                'variations': []
            }
            
            processed.add(i)
            
            # Find similar recommendations
            for j, rec2 in enumerate(recommendations[i+1:], i+1):
                if j in processed:
                    continue
                
                if self._are_recommendations_similar(rec1['text'], rec2['text']):
                    group['supporting_domains'].append(rec2['source_domain'])
                    group['variations'].append(rec2['text'])
                    processed.add(j)
                    
                    # Upgrade priority if any expert says it's high
                    if rec2['priority'] == 'high':
                        group['priority'] = 'high'
            
            grouped.append(group)
        
        return grouped
    
    def _are_recommendations_similar(self, rec1: str, rec2: str) -> bool:
        """Check if two recommendations are similar"""
        # Simple keyword-based similarity
        words1 = set(rec1.lower().split())
        words2 = set(rec2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return False
        
        similarity = intersection / union
        return similarity > 0.4  # 40% similarity threshold
    
    def get_expert_status(self) -> Dict[str, Any]:
        """Get status of expert analysis capabilities"""
        return {
            'available': self.available,
            'model': self.model if self.available else None,
            'max_tokens': self.max_tokens,
            'expert_domains': list(self.expert_personas.keys()),
            'advanced_features': {
                'multi_expert_consultation': True,
                'expert_synthesis': True,
                'domain_specialization': True,
                'structured_analysis': True
            }
        }
