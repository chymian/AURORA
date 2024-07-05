tools = [
    {
        "type": "function",
        "function": {
            "name": "run_local_command",
            "description": "Execute a local command on the system to perform tasks such as file manipulation, retrieving system information, or running scripts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The specific command to execute on the local system.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_research",
            "description": "Perform a web research query to gather information from online sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research query to perform.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Analyze an image from a provided URL or a local path and generate a description of the image's content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": "The URL or local path of the image to analyze.",
                    }
                },
                "required": ["image_url"],
            },
        },
    },


        {
        "type": "function",
        "function": {
            "name": "call_expert",
            "description": "A tool that can ask an expert in any field by providing the expertise and the question. The expert will answer the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expertise": {
                        "type": "string",
                        "description": "The expertise of the expert you need. IE: math, science, etc.",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question you want to ask the expert.",
                    },
                },
                "required": ["expertise", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_directory_manager",
            "description": "Perform file and directory operations on the local system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list_directory", "create_directory", "delete_item", "move_item", "copy_item", "read_file", "write_file", "search_files", "get_file_info"],
                        "description": "The action to perform on files or directories.",
                    },
                    "path": {
                        "type": "string",
                        "description": "The path to the file or directory.",
                    },
                    "source": {
                        "type": "string",
                        "description": "The source path for move or copy operations.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "The destination path for move or copy operations.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to a file.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "The search pattern for finding files.",
                    },
                },
                "required": ["action"],
            },
        },
    },
]
