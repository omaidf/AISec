import os
import re
import json
import argparse
from dataclasses import dataclass, field, asdict

@dataclass
class Endpoint:
    url: str
    http_method: str
    file_path: str
    line_number: int
    parameters: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

class EndpointFinder:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.endpoints = []
        self.csharp_extensions = ('.cs',)
        
        # Patterns for different endpoint declaration styles
        self.patterns = [
            # ASP.NET Core attributes
            r'\[(?:Http(?:Get|Post|Put|Delete|Patch)|Route)\s*\("([^"]*)"\)\]',
            r'\[(?:Http(?:Get|Post|Put|Delete|Patch)|Route)\s*\((?:name\s*:\s*)?"([^"]*)"\)\]',
            
            # MapGet/MapPost style
            r'app\.Map(?:Get|Post|Put|Delete|Patch)\s*\("([^"]*)"',
            
            # Minimal API method style
            r'\.(?:WithName\s*\(\s*"[^"]*"\s*\)\s*)?\.(?:Accepts|Produces)\s*\([^)]*\)\s*\.\s*Executes\s*\([^)]*\)'
        ]
        
        # HTTP method patterns
        self.http_method_patterns = {
            'GET': [r'\[HttpGet', r'app\.MapGet', r'\.get\('],
            'POST': [r'\[HttpPost', r'app\.MapPost', r'\.post\('],
            'PUT': [r'\[HttpPut', r'app\.MapPut', r'\.put\('],
            'DELETE': [r'\[HttpDelete', r'app\.MapDelete', r'\.delete\('],
            'PATCH': [r'\[HttpPatch', r'app\.MapPatch', r'\.patch\('],
        }
        
        # Parameter patterns
        self.parameter_patterns = [
            # Method parameters
            r'public\s+(?:async\s+)?(?:\w+)\s+\w+\s*\(([^)]*)\)',
            # FromBody, FromQuery, etc.
            r'\[(?:FromBody|FromQuery|FromRoute|FromForm|FromHeader)\]\s+(?:\w+)\s+(\w+)',
            # Parameter declaration in minimal APIs
            r'app\.Map\w+\([^,]+,\s*\(([^)]*)\)\s*=>'
        ]

    def find_csharp_files(self):
        """Generator to find C# files in repository"""
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.lower().endswith(self.csharp_extensions):
                    yield os.path.join(root, file)

    def extract_http_method(self, line):
        """Extract HTTP method from a line of code"""
        for method, patterns in self.http_method_patterns.items():
            if any(re.search(p, line, re.IGNORECASE) for p in patterns):
                return method
        
        # Default to GET if specific method not found but it's a route
        if re.search(r'\[Route', line, re.IGNORECASE) or re.search(r'app\.Map', line, re.IGNORECASE):
            return "GET"
        
        return "UNKNOWN"

    def extract_parameters(self, file_content, line_index):
        """Extract parameters from the endpoint method"""
        parameters = []
        
        # Look at the next few lines for parameter information
        context_range = min(10, len(file_content) - line_index)
        context = '\n'.join(file_content[line_index:line_index + context_range])
        
        for pattern in self.parameter_patterns:
            param_matches = re.findall(pattern, context)
            for param_match in param_matches:
                if isinstance(param_match, tuple):
                    param_match = param_match[0]
                
                # Split and clean each parameter
                if param_match:
                    param_parts = [p.strip() for p in param_match.split(',')]
                    for part in param_parts:
                        if part:
                            # Extract parameter name and type
                            param_info = part.split()
                            if len(param_info) >= 2:
                                param_name = param_info[-1].replace("[", "").replace("]", "")
                                param_type = param_info[-2]
                                parameters.append({"name": param_name, "type": param_type})
        
        return parameters

    def find_endpoints(self):
        """Find all API endpoints in the repository"""
        for file_path in self.find_csharp_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.readlines()
                
                # Join lines to handle multi-line declarations
                full_content = ''.join(content)
                file_content = content
                
                for i, line in enumerate(content):
                    for pattern in self.patterns:
                        matches = re.findall(pattern, line)
                        if matches:
                            for match in matches:
                                http_method = self.extract_http_method(line)
                                parameters = self.extract_parameters(file_content, i)
                                
                                # Clean up the URL
                                url = match.strip()
                                
                                # Add endpoint
                                self.endpoints.append(Endpoint(
                                    url=url,
                                    http_method=http_method,
                                    file_path=file_path,
                                    line_number=i + 1,
                                    parameters=parameters
                                ))
                                
                # Handle controller-based routing
                self.handle_controller_routing(file_path, full_content)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    def handle_controller_routing(self, file_path, content):
        """Handle controller-based routing with route prefix"""
        # Find controller route prefix
        controller_match = re.search(r'\[Route\("([^"]*)"\)\]\s*public\s+class\s+\w+Controller', content)
        if not controller_match:
            return
            
        base_route = controller_match.group(1).rstrip('/')
        
        # Find all action methods with routes in this controller
        action_matches = re.finditer(r'\[Http(?:Get|Post|Put|Delete|Patch)\("([^"]*)"\)\]\s*public\s+(?:async\s+)?(?:\w+)\s+(\w+)\s*\(([^)]*)\)', content)
        
        for match in action_matches:
            action_route = match.group(1).lstrip('/')
            full_route = f"{base_route}/{action_route}" if action_route else base_route
            
            # Get line number
            line_number = content[:match.start()].count('\n') + 1
            
            # Extract parameters
            param_str = match.group(3)
            parameters = []
            if param_str:
                params = [p.strip() for p in param_str.split(',')]
                for p in params:
                    parts = p.split()
                    if len(parts) >= 2:
                        param_name = parts[-1]
                        param_type = ' '.join(parts[:-1])
                        parameters.append({"name": param_name, "type": param_type})
            
            # Determine HTTP method
            http_method_match = re.search(r'Http(Get|Post|Put|Delete|Patch)', match.group(0))
            http_method = http_method_match.group(1).upper() if http_method_match else "UNKNOWN"
            
            self.endpoints.append(Endpoint(
                url=full_route,
                http_method=http_method,
                file_path=file_path,
                line_number=line_number,
                parameters=parameters
            ))

    def generate_report(self, output_file):
        """Generate a JSON report of all endpoints"""
        # Sort endpoints by URL for better readability
        self.endpoints.sort(key=lambda x: x.url)
        
        # Convert to dictionary
        endpoints_dict = [endpoint.to_dict() for endpoint in self.endpoints]
        
        with open(output_file, 'w') as f:
            json.dump(endpoints_dict, f, indent=2)
        
        print(f"Found {len(self.endpoints)} endpoints. Report generated: {output_file}")
        
        # Print summary
        print("\nEndpoint Summary:")
        for endpoint in self.endpoints:
            print(f"{endpoint.http_method} {endpoint.url} ({endpoint.file_path}:{endpoint.line_number})")
            if endpoint.parameters:
                print(f"  Parameters: {', '.join([p['name'] for p in endpoint.parameters])}")

def main():
    parser = argparse.ArgumentParser(description="Find API endpoints in a C# repository")
    parser.add_argument("repo_path", help="Path to C# repository")
    parser.add_argument("-o", "--output", default="endpoints.json", help="Output JSON file")
    args = parser.parse_args()
    
    finder = EndpointFinder(args.repo_path)
    finder.find_endpoints()
    finder.generate_report(args.output)

if __name__ == "__main__":
    main()