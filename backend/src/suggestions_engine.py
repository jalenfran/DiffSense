"""
Smart Suggestions Engine - Contextual and Actionable Recommendations
Generates highly relevant, prioritized suggestions based on code analysis and domain expertise
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Category(Enum):
    SECURITY = "security"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"
    QUALITY = "quality"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"

@dataclass
class Suggestion:
    """A smart, actionable suggestion with context"""
    title: str
    description: str
    priority: Priority
    category: Category
    impact: str  # business impact description
    effort: str  # implementation effort estimate
    actionable_steps: List[str]
    code_examples: List[str]
    tools_recommended: List[str]
    files_affected: List[str]
    success_metrics: List[str]
    risk_if_ignored: str
    confidence: float
    reasoning: str

class SuggestionsEngine:
    """Generates intelligent, contextual suggestions based on code analysis"""
    
    def __init__(self):
        # Pattern-based suggestion templates
        self.suggestion_patterns = {
            'security': {
                'hardcoded_secrets': {
                    'title': 'Remove Hardcoded Secrets',
                    'description': 'Hardcoded API keys, passwords, or tokens detected',
                    'priority': Priority.CRITICAL,
                    'impact': 'Prevents security breaches and unauthorized access',
                    'effort': '2-4 hours per file',
                    'risk_if_ignored': 'Data breach, unauthorized system access, compliance violations'
                },
                'sql_injection': {
                    'title': 'Fix SQL Injection Vulnerabilities',
                    'description': 'Dynamic SQL construction without parameterization detected',
                    'priority': Priority.CRITICAL,
                    'impact': 'Prevents database attacks and data compromise',
                    'effort': '4-8 hours per vulnerable query',
                    'risk_if_ignored': 'Database compromise, data theft, system corruption'
                },
                'auth_weaknesses': {
                    'title': 'Strengthen Authentication Implementation',
                    'description': 'Weak authentication patterns or missing security controls',
                    'priority': Priority.HIGH,
                    'impact': 'Improves user security and system integrity',
                    'effort': '1-2 days',
                    'risk_if_ignored': 'Unauthorized access, account takeover, privilege escalation'
                }
            },
            'performance': {
                'high_complexity': {
                    'title': 'Optimize High-Complexity Functions',
                    'description': 'Functions with high cyclomatic complexity need optimization',
                    'priority': Priority.MEDIUM,
                    'impact': 'Improves response times and system scalability',
                    'effort': '1-3 days per function',
                    'risk_if_ignored': 'Performance degradation, poor user experience, scaling issues'
                },
                'n_plus_one': {
                    'title': 'Fix N+1 Query Problems',
                    'description': 'Database queries in loops causing performance issues',
                    'priority': Priority.HIGH,
                    'impact': 'Dramatically improves database performance',
                    'effort': '4-8 hours per occurrence',
                    'risk_if_ignored': 'Database overload, slow response times, system crashes'
                },
                'inefficient_algorithms': {
                    'title': 'Optimize Algorithm Efficiency',
                    'description': 'Inefficient algorithms or data structures detected',
                    'priority': Priority.MEDIUM,
                    'impact': 'Reduces CPU usage and improves response times',
                    'effort': '1-2 days',
                    'risk_if_ignored': 'High resource consumption, poor scalability'
                }
            },
            'architecture': {
                'tight_coupling': {
                    'title': 'Reduce Component Coupling',
                    'description': 'High coupling between components affecting maintainability',
                    'priority': Priority.MEDIUM,
                    'impact': 'Improves code maintainability and testability',
                    'effort': '1-2 weeks',
                    'risk_if_ignored': 'Difficult maintenance, poor testability, brittle system'
                },
                'missing_patterns': {
                    'title': 'Implement Design Patterns',
                    'description': 'Code could benefit from established design patterns',
                    'priority': Priority.LOW,
                    'impact': 'Improves code organization and maintainability',
                    'effort': '3-5 days',
                    'risk_if_ignored': 'Poor code organization, difficult to extend'
                },
                'layer_violations': {
                    'title': 'Fix Architecture Layer Violations',
                    'description': 'Components violating architectural boundaries',
                    'priority': Priority.HIGH,
                    'impact': 'Maintains clean architecture and separation of concerns',
                    'effort': '1-2 weeks',
                    'risk_if_ignored': 'Architecture degradation, difficult maintenance'
                }
            },
            'quality': {
                'technical_debt': {
                    'title': 'Address Technical Debt',
                    'description': 'TODO, FIXME, and HACK comments indicate technical debt',
                    'priority': Priority.MEDIUM,
                    'impact': 'Improves code quality and reduces future maintenance costs',
                    'effort': '1-2 days per issue',
                    'risk_if_ignored': 'Accumulating technical debt, increasing maintenance costs'
                },
                'code_duplication': {
                    'title': 'Eliminate Code Duplication',
                    'description': 'Duplicated code blocks detected across files',
                    'priority': Priority.MEDIUM,
                    'impact': 'Reduces maintenance burden and improves consistency',
                    'effort': '4-8 hours per duplication',
                    'risk_if_ignored': 'Inconsistent behavior, difficult maintenance'
                },
                'long_methods': {
                    'title': 'Break Down Large Methods',
                    'description': 'Methods are too long and complex',
                    'priority': Priority.LOW,
                    'impact': 'Improves readability and testability',
                    'effort': '2-4 hours per method',
                    'risk_if_ignored': 'Poor readability, difficult debugging'
                }
            }
        }
        
        # Technology-specific suggestions
        self.tech_specific_suggestions = {
            'python': {
                'async_opportunities': 'Consider using async/await for I/O operations',
                'typing_missing': 'Add type hints for better code documentation',
                'list_comprehensions': 'Use list comprehensions for better performance'
            },
            'javascript': {
                'promise_chains': 'Consider using async/await instead of promise chains',
                'var_usage': 'Replace var with let/const for better scoping',
                'arrow_functions': 'Use arrow functions for cleaner syntax'
            },
            'java': {
                'stream_api': 'Use Stream API for functional programming',
                'optional_usage': 'Use Optional to handle null values safely',
                'lambda_expressions': 'Use lambda expressions for cleaner code'
            }
        }
        
        # Industry best practices by domain
        self.best_practices = {
            'security': [
                'Implement defense in depth',
                'Use principle of least privilege',
                'Regular security audits',
                'Secure coding standards',
                'Input validation and sanitization'
            ],
            'performance': [
                'Profile before optimizing',
                'Cache frequently accessed data',
                'Minimize database queries',
                'Use efficient algorithms',
                'Monitor performance metrics'
            ],
            'architecture': [
                'Follow SOLID principles',
                'Maintain loose coupling',
                'Design for testability',
                'Use appropriate design patterns',
                'Document architectural decisions'
            ],
            'quality': [
                'Write self-documenting code',
                'Maintain consistent style',
                'Regular refactoring',
                'Comprehensive testing',
                'Code review processes'
            ]
        }
    
    def generate_smart_suggestions(self, query: str, domain: str, code_analysis: Dict[str, Any], 
                                  file_analyses: List[Any], expert_analysis: Optional[Any] = None) -> List[Suggestion]:
        """Generate smart, contextual suggestions based on analysis"""
        
        suggestions = []
        
        # Generate pattern-based suggestions
        pattern_suggestions = self._generate_pattern_suggestions(domain, code_analysis, file_analyses)
        suggestions.extend(pattern_suggestions)
        
        # Generate context-aware suggestions
        context_suggestions = self._generate_context_suggestions(query, domain, code_analysis)
        suggestions.extend(context_suggestions)
        
        # Generate technology-specific suggestions
        tech_suggestions = self._generate_tech_suggestions(file_analyses)
        suggestions.extend(tech_suggestions)
        
        # Generate expert-driven suggestions
        if expert_analysis:
            expert_suggestions = self._generate_expert_suggestions(expert_analysis)
            suggestions.extend(expert_suggestions)
        
        # Generate proactive suggestions
        proactive_suggestions = self._generate_proactive_suggestions(domain, file_analyses)
        suggestions.extend(proactive_suggestions)
        
        # Prioritize and deduplicate
        final_suggestions = self._prioritize_and_deduplicate(suggestions)
        
        return final_suggestions[:8]  # Return top 8 suggestions
    
    def _generate_pattern_suggestions(self, domain: str, code_analysis: Dict[str, Any], 
                                    file_analyses: List[Any]) -> List[Suggestion]:
        """Generate suggestions based on detected patterns"""
        suggestions = []
        
        if domain == 'security':
            # Security pattern-based suggestions
            security_data = code_analysis.get('security_analysis', {})
            issues = security_data.get('issues_found', [])
            
            for issue in issues:
                suggestion = self._create_security_suggestion(issue, security_data.get('affected_files', []))
                if suggestion:
                    suggestions.append(suggestion)
        
        elif domain == 'performance':
            # Performance pattern-based suggestions
            perf_data = code_analysis.get('performance_analysis', {})
            bottlenecks = perf_data.get('potential_bottlenecks', [])
            
            for file_path, complexity in bottlenecks:
                suggestion = self._create_performance_suggestion(file_path, complexity)
                suggestions.append(suggestion)
        
        elif domain == 'architecture':
            # Architecture pattern-based suggestions
            arch_data = code_analysis.get('architecture_analysis', {})
            
            if not arch_data.get('patterns_detected'):
                suggestions.append(self._create_architecture_pattern_suggestion())
            
            bottlenecks = arch_data.get('potential_bottlenecks', [])
            if bottlenecks:
                suggestion = self._create_coupling_suggestion(bottlenecks)
                suggestions.append(suggestion)
        
        elif domain == 'quality':
            # Quality pattern-based suggestions
            quality_data = code_analysis.get('quality_analysis', {})
            
            high_complexity = quality_data.get('high_complexity_files', [])
            for file_path, complexity in high_complexity:
                suggestion = self._create_complexity_suggestion(file_path, complexity)
                suggestions.append(suggestion)
            
            tech_debt = quality_data.get('technical_debt_files', [])
            for file_path, debt_indicators in tech_debt:
                suggestion = self._create_tech_debt_suggestion(file_path, debt_indicators)
                suggestions.append(suggestion)
        
        return suggestions
    
    def _create_security_suggestion(self, issue: str, affected_files: List[str]) -> Optional[Suggestion]:
        """Create security-specific suggestion"""
        
        if 'password' in issue.lower() or 'secret' in issue.lower() or 'api_key' in issue.lower():
            template = self.suggestion_patterns['security']['hardcoded_secrets']
            
            return Suggestion(
                title=template['title'],
                description=f"{template['description']}: {issue}",
                priority=template['priority'],
                category=Category.SECURITY,
                impact=template['impact'],
                effort=template['effort'],
                actionable_steps=[
                    "1. Move secrets to environment variables or secure vault",
                    "2. Update code to read from secure configuration",
                    "3. Add secrets scanning to CI/CD pipeline",
                    "4. Rotate any exposed credentials immediately"
                ],
                code_examples=[
                    "# Instead of: api_key = 'abc123'",
                    "# Use: api_key = os.getenv('API_KEY')"
                ],
                tools_recommended=['HashiCorp Vault', 'AWS Secrets Manager', 'Git-secrets'],
                files_affected=affected_files[:5],
                success_metrics=['Zero hardcoded secrets in codebase', 'Passing security scans'],
                risk_if_ignored=template['risk_if_ignored'],
                confidence=0.9,
                reasoning="Hardcoded secrets are a critical security vulnerability"
            )
        
        elif 'sql' in issue.lower() and 'injection' in issue.lower():
            template = self.suggestion_patterns['security']['sql_injection']
            
            return Suggestion(
                title=template['title'],
                description=f"{template['description']}: {issue}",
                priority=template['priority'],
                category=Category.SECURITY,
                impact=template['impact'],
                effort=template['effort'],
                actionable_steps=[
                    "1. Replace string concatenation with parameterized queries",
                    "2. Use ORM framework if not already implemented",
                    "3. Implement input validation and sanitization",
                    "4. Add SQL injection testing to security suite"
                ],
                code_examples=[
                    "# Instead of: f'SELECT * FROM users WHERE id = {user_id}'",
                    "# Use: cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))"
                ],
                tools_recommended=['SQLAlchemy', 'Prepared Statements', 'OWASP ZAP'],
                files_affected=affected_files[:5],
                success_metrics=['Zero SQL injection vulnerabilities', 'Parameterized queries usage'],
                risk_if_ignored=template['risk_if_ignored'],
                confidence=0.95,
                reasoning="SQL injection is a critical OWASP Top 10 vulnerability"
            )
        
        return None
    
    def _create_performance_suggestion(self, file_path: str, complexity: float) -> Suggestion:
        """Create performance-specific suggestion"""
        template = self.suggestion_patterns['performance']['high_complexity']
        
        return Suggestion(
            title=f"Optimize {file_path.split('/')[-1]}",
            description=f"High complexity score ({complexity:.1f}) indicates performance optimization opportunity",
            priority=Priority.HIGH if complexity > 7 else Priority.MEDIUM,
            category=Category.PERFORMANCE,
            impact=template['impact'],
            effort=template['effort'],
            actionable_steps=[
                f"1. Profile {file_path} to identify bottlenecks",
                "2. Break down complex functions into smaller units",
                "3. Optimize algorithms and data structures",
                "4. Add performance monitoring"
            ],
            code_examples=[
                "# Use profiling to identify hotspots",
                "# Consider caching for expensive operations",
                "# Optimize database queries and data access"
            ],
            tools_recommended=['cProfile', 'py-spy', 'New Relic', 'DataDog'],
            files_affected=[file_path],
            success_metrics=[f'Complexity score < 5', 'Response time improvement'],
            risk_if_ignored=template['risk_if_ignored'],
            confidence=0.8,
            reasoning=f"High complexity ({complexity:.1f}) correlates with performance issues"
        )
    
    def _create_architecture_pattern_suggestion(self) -> Suggestion:
        """Create architecture pattern suggestion"""
        template = self.suggestion_patterns['architecture']['missing_patterns']
        
        return Suggestion(
            title="Implement Architectural Patterns",
            description="Code structure could benefit from established design patterns",
            priority=template['priority'],
            category=Category.ARCHITECTURE,
            impact=template['impact'],
            effort=template['effort'],
            actionable_steps=[
                "1. Analyze current code organization",
                "2. Identify areas that would benefit from patterns",
                "3. Implement appropriate patterns (Factory, Observer, Strategy)",
                "4. Document architectural decisions"
            ],
            code_examples=[
                "# Consider Factory pattern for object creation",
                "# Use Observer pattern for event handling",
                "# Implement Strategy pattern for algorithms"
            ],
            tools_recommended=['UML tools', 'Architecture documentation', 'Design pattern libraries'],
            files_affected=[],
            success_metrics=['Clear architectural boundaries', 'Improved code organization'],
            risk_if_ignored=template['risk_if_ignored'],
            confidence=0.7,
            reasoning="No clear architectural patterns detected in codebase"
        )
    
    def _create_coupling_suggestion(self, bottlenecks: List[str]) -> Suggestion:
        """Create coupling reduction suggestion"""
        template = self.suggestion_patterns['architecture']['tight_coupling']
        
        return Suggestion(
            title="Reduce Component Coupling",
            description=f"High coupling detected in {len(bottlenecks)} components",
            priority=template['priority'],
            category=Category.ARCHITECTURE,
            impact=template['impact'],
            effort=template['effort'],
            actionable_steps=[
                "1. Identify tightly coupled components",
                "2. Introduce interfaces and dependency injection",
                "3. Apply separation of concerns principle",
                "4. Refactor to reduce interdependencies"
            ],
            code_examples=[
                "# Use dependency injection",
                "# Implement interfaces for loose coupling",
                "# Apply Single Responsibility Principle"
            ],
            tools_recommended=['Dependency injection frameworks', 'Architecture analysis tools'],
            files_affected=bottlenecks[:5],
            success_metrics=['Reduced coupling metrics', 'Improved testability'],
            risk_if_ignored=template['risk_if_ignored'],
            confidence=0.8,
            reasoning="High coupling detected in architecture analysis"
        )
    
    def _create_complexity_suggestion(self, file_path: str, complexity: float) -> Suggestion:
        """Create complexity reduction suggestion"""
        template = self.suggestion_patterns['quality']['long_methods']
        
        return Suggestion(
            title=f"Refactor Complex Code in {file_path.split('/')[-1]}",
            description=f"Complexity score {complexity:.1f} exceeds recommended threshold",
            priority=Priority.HIGH if complexity > 8 else Priority.MEDIUM,
            category=Category.QUALITY,
            impact=template['impact'],
            effort=template['effort'],
            actionable_steps=[
                f"1. Analyze complex functions in {file_path}",
                "2. Break down large methods into smaller functions",
                "3. Extract common functionality",
                "4. Add unit tests for refactored code"
            ],
            code_examples=[
                "# Extract methods to reduce complexity",
                "# Use early returns to simplify logic",
                "# Apply Single Responsibility Principle"
            ],
            tools_recommended=['SonarQube', 'Code complexity analyzers', 'Refactoring tools'],
            files_affected=[file_path],
            success_metrics=['Complexity score < 5', 'Improved readability'],
            risk_if_ignored=template['risk_if_ignored'],
            confidence=0.85,
            reasoning=f"Complexity score {complexity:.1f} is above recommended threshold"
        )
    
    def _create_tech_debt_suggestion(self, file_path: str, debt_indicators: List[str]) -> Suggestion:
        """Create technical debt suggestion"""
        template = self.suggestion_patterns['quality']['technical_debt']
        
        return Suggestion(
            title=f"Address Technical Debt in {file_path.split('/')[-1]}",
            description=f"Technical debt markers found: {', '.join(debt_indicators[:3])}",
            priority=Priority.MEDIUM,
            category=Category.MAINTENANCE,
            impact=template['impact'],
            effort=template['effort'],
            actionable_steps=[
                f"1. Review and prioritize debt items in {file_path}",
                "2. Create specific tasks for each TODO/FIXME",
                "3. Schedule regular debt cleanup sprints",
                "4. Implement coding standards to prevent future debt"
            ],
            code_examples=[
                "# Replace TODO comments with proper implementation",
                "# Fix FIXME items with proper solutions",
                "# Remove HACK implementations"
            ],
            tools_recommended=['Issue tracking systems', 'Code review tools', 'Static analysis'],
            files_affected=[file_path],
            success_metrics=['Reduced TODO/FIXME count', 'Improved code quality metrics'],
            risk_if_ignored=template['risk_if_ignored'],
            confidence=0.9,
            reasoning=f"Technical debt indicators detected: {', '.join(debt_indicators)}"
        )
    
    def _generate_context_suggestions(self, query: str, domain: str, code_analysis: Dict[str, Any]) -> List[Suggestion]:
        """Generate suggestions based on query context"""
        suggestions = []
        query_lower = query.lower()
        
        # Query-specific suggestions
        if 'test' in query_lower or 'testing' in query_lower:
            suggestions.append(self._create_testing_suggestion(code_analysis))
        
        if 'deploy' in query_lower or 'deployment' in query_lower:
            suggestions.append(self._create_deployment_suggestion())
        
        if 'monitor' in query_lower or 'monitoring' in query_lower:
            suggestions.append(self._create_monitoring_suggestion(domain))
        
        if 'document' in query_lower or 'documentation' in query_lower:
            suggestions.append(self._create_documentation_suggestion())
        
        return suggestions
    
    def _create_testing_suggestion(self, code_analysis: Dict[str, Any]) -> Suggestion:
        """Create testing-related suggestion"""
        return Suggestion(
            title="Improve Test Coverage",
            description="Enhance testing strategy for better code quality",
            priority=Priority.MEDIUM,
            category=Category.TESTING,
            impact="Reduces bugs, improves confidence in deployments",
            effort="1-2 weeks",
            actionable_steps=[
                "1. Analyze current test coverage",
                "2. Identify untested critical paths",
                "3. Implement unit tests for core functionality",
                "4. Add integration tests for key workflows",
                "5. Set up automated testing in CI/CD"
            ],
            code_examples=[
                "# Add unit tests for business logic",
                "# Implement integration tests",
                "# Use mocking for external dependencies"
            ],
            tools_recommended=['pytest', 'Jest', 'JUnit', 'Coverage.py', 'Cypress'],
            files_affected=[],
            success_metrics=['Test coverage > 80%', 'Reduced bug reports'],
            risk_if_ignored="Increased bugs, difficult refactoring, production issues",
            confidence=0.8,
            reasoning="Testing mentioned in query, suggesting testing improvements needed"
        )
    
    def _create_deployment_suggestion(self) -> Suggestion:
        """Create deployment-related suggestion"""
        return Suggestion(
            title="Optimize Deployment Pipeline",
            description="Improve deployment process for reliability and speed",
            priority=Priority.MEDIUM,
            category=Category.DEPLOYMENT,
            impact="Faster, more reliable deployments with reduced downtime",
            effort="1-2 weeks",
            actionable_steps=[
                "1. Implement infrastructure as code",
                "2. Set up automated deployment pipeline",
                "3. Add deployment health checks",
                "4. Implement blue-green or canary deployments",
                "5. Set up monitoring and alerting"
            ],
            code_examples=[
                "# Use Docker for containerization",
                "# Implement Kubernetes manifests",
                "# Set up CI/CD with GitHub Actions or Jenkins"
            ],
            tools_recommended=['Docker', 'Kubernetes', 'Terraform', 'GitHub Actions', 'Jenkins'],
            files_affected=[],
            success_metrics=['Deployment time < 10 minutes', 'Zero-downtime deployments'],
            risk_if_ignored="Manual deployments, higher error rates, longer downtime",
            confidence=0.7,
            reasoning="Deployment mentioned in query, suggesting deployment improvements needed"
        )
    
    def _create_monitoring_suggestion(self, domain: str) -> Suggestion:
        """Create monitoring-related suggestion"""
        
        domain_specific_metrics = {
            'security': ['Failed login attempts', 'Security scan results', 'Vulnerability counts'],
            'performance': ['Response times', 'Throughput', 'Resource utilization'],
            'quality': ['Code coverage', 'Technical debt ratio', 'Bug rates'],
            'architecture': ['Component dependencies', 'Service health', 'Architecture compliance']
        }
        
        metrics = domain_specific_metrics.get(domain, ['System health', 'Error rates', 'Performance metrics'])
        
        return Suggestion(
            title=f"Implement {domain.title()} Monitoring",
            description=f"Set up comprehensive monitoring for {domain} aspects",
            priority=Priority.MEDIUM,
            category=Category.DEPLOYMENT,
            impact="Proactive issue detection and faster resolution",
            effort="1 week",
            actionable_steps=[
                "1. Define key metrics to monitor",
                "2. Set up monitoring infrastructure",
                "3. Create dashboards for visualization",
                "4. Configure alerting rules",
                "5. Establish incident response procedures"
            ],
            code_examples=[
                "# Add application metrics",
                "# Implement health check endpoints",
                "# Set up structured logging"
            ],
            tools_recommended=['Prometheus', 'Grafana', 'DataDog', 'New Relic', 'Splunk'],
            files_affected=[],
            success_metrics=metrics,
            risk_if_ignored="Blind spots in system health, reactive issue resolution",
            confidence=0.7,
            reasoning=f"Monitoring mentioned in query for {domain} domain"
        )
    
    def _create_documentation_suggestion(self) -> Suggestion:
        """Create documentation-related suggestion"""
        return Suggestion(
            title="Improve Code Documentation",
            description="Enhance documentation for better maintainability",
            priority=Priority.LOW,
            category=Category.DOCUMENTATION,
            impact="Improved developer onboarding and code understanding",
            effort="1-2 weeks",
            actionable_steps=[
                "1. Audit existing documentation coverage",
                "2. Document APIs and interfaces",
                "3. Add inline code comments for complex logic",
                "4. Create architectural decision records",
                "5. Set up automated documentation generation"
            ],
            code_examples=[
                "# Add docstrings to functions",
                "# Document API endpoints",
                "# Create README files for modules"
            ],
            tools_recommended=['Sphinx', 'JSDoc', 'Swagger', 'GitBook', 'Confluence'],
            files_affected=[],
            success_metrics=['API documentation coverage > 90%', 'Reduced onboarding time'],
            risk_if_ignored="Difficult maintenance, slow onboarding, knowledge silos",
            confidence=0.6,
            reasoning="Documentation mentioned in query, suggesting documentation improvements needed"
        )
    
    def _generate_tech_suggestions(self, file_analyses: List[Any]) -> List[Suggestion]:
        """Generate technology-specific suggestions"""
        suggestions = []
        
        # Analyze predominant languages
        languages = {}
        for analysis in file_analyses:
            lang = analysis.language
            if lang in languages:
                languages[lang] += 1
            else:
                languages[lang] = 1
        
        # Generate suggestions for main language
        if languages:
            main_language = max(languages, key=languages.get)
            tech_suggestions = self.tech_specific_suggestions.get(main_language, {})
            
            for pattern, description in tech_suggestions.items():
                suggestion = Suggestion(
                    title=f"Modernize {main_language.title()} Code",
                    description=description,
                    priority=Priority.LOW,
                    category=Category.QUALITY,
                    impact="Improved code quality and maintainability",
                    effort="2-4 hours per file",
                    actionable_steps=[
                        f"1. Review {main_language} best practices",
                        "2. Update code to use modern language features",
                        "3. Run automated code formatters",
                        "4. Add linting rules for consistency"
                    ],
                    code_examples=[description],
                    tools_recommended=[f'{main_language} linters', 'Code formatters', 'Style guides'],
                    files_affected=[],
                    success_metrics=['Consistent code style', 'Modern language usage'],
                    risk_if_ignored="Outdated code patterns, inconsistent style",
                    confidence=0.6,
                    reasoning=f"Technology-specific improvements for {main_language}"
                )
                suggestions.append(suggestion)
                break  # Only add one tech suggestion to avoid spam
        
        return suggestions
    
    def _generate_expert_suggestions(self, expert_analysis: Any) -> List[Suggestion]:
        """Generate suggestions from expert analysis"""
        suggestions = []
        
        if hasattr(expert_analysis, 'actionable_recommendations'):
            for rec in expert_analysis.actionable_recommendations[:3]:  # Top 3 expert recommendations
                suggestion = Suggestion(
                    title=f"Expert Recommendation: {rec.get('text', '')[:50]}...",
                    description=rec.get('text', ''),
                    priority=Priority(rec.get('priority', 'medium')),
                    category=Category(rec.get('category', 'quality')),
                    impact="Expert-identified improvement opportunity",
                    effort="As recommended by domain expert",
                    actionable_steps=[rec.get('text', '')],
                    code_examples=[],
                    tools_recommended=[],
                    files_affected=[],
                    success_metrics=['Expert-defined success criteria'],
                    risk_if_ignored="Expert-identified risks",
                    confidence=0.9,
                    reasoning="Recommendation from domain expert analysis"
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_proactive_suggestions(self, domain: str, file_analyses: List[Any]) -> List[Suggestion]:
        """Generate proactive suggestions to prevent future issues"""
        suggestions = []
        
        # Best practices for the domain
        practices = self.best_practices.get(domain, [])
        
        if practices:
            practice = practices[0]  # Pick one practice to suggest
            
            suggestion = Suggestion(
                title=f"Implement {domain.title()} Best Practice",
                description=f"Proactive improvement: {practice}",
                priority=Priority.LOW,
                category=Category(domain),
                impact="Prevents future issues and improves code quality",
                effort="1-2 days",
                actionable_steps=[
                    f"1. Research {practice} implementation",
                    "2. Create implementation plan",
                    "3. Apply practice to codebase",
                    "4. Document the practice for team"
                ],
                code_examples=[],
                tools_recommended=[],
                files_affected=[],
                success_metrics=[f'{practice} implemented across codebase'],
                risk_if_ignored=f"Missing {domain} best practice may lead to future issues",
                confidence=0.5,
                reasoning=f"Proactive suggestion for {domain} best practices"
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _prioritize_and_deduplicate(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """Prioritize suggestions and remove duplicates"""
        
        # Remove duplicates based on title similarity
        unique_suggestions = []
        seen_titles = set()
        
        for suggestion in suggestions:
            title_key = suggestion.title.lower().replace(' ', '')
            if title_key not in seen_titles:
                unique_suggestions.append(suggestion)
                seen_titles.add(title_key)
        
        # Sort by priority and confidence
        priority_order = {
            Priority.CRITICAL: 4,
            Priority.HIGH: 3,
            Priority.MEDIUM: 2,
            Priority.LOW: 1
        }
        
        unique_suggestions.sort(
            key=lambda x: (priority_order[x.priority], x.confidence),
            reverse=True
        )
        
        return unique_suggestions
    
    def get_suggestion_summary(self, suggestions: List[Suggestion]) -> Dict[str, Any]:
        """Get summary of suggestions for reporting"""
        
        summary = {
            'total_suggestions': len(suggestions),
            'by_priority': {},
            'by_category': {},
            'estimated_effort': {},
            'top_risks': []
        }
        
        for suggestion in suggestions:
            # Count by priority
            priority = suggestion.priority.value
            summary['by_priority'][priority] = summary['by_priority'].get(priority, 0) + 1
            
            # Count by category
            category = suggestion.category.value
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            # Collect high-priority risks
            if suggestion.priority in [Priority.CRITICAL, Priority.HIGH]:
                summary['top_risks'].append({
                    'title': suggestion.title,
                    'risk': suggestion.risk_if_ignored
                })
        
        return summary
