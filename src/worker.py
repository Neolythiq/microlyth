from abc import ABC, abstractmethod
from typing import Callable, List

import xml.etree.ElementTree as ET
from xml.dom import minidom

class AgentBase(ABC):
    @abstractmethod
    def Manifest(self):
        pass

    def Tools(self):
        return self.tools

    def ToolsManifest(self):
        manifest = ET.Element("tools")
        for toolName in self.tools.keys():
            tool = ET.SubElement(manifest, "tool")
            ET.SubElement(tool, "toolName").text = toolName
            ET.SubElement(tool, "toolDescription").text = self.tools[toolName].__doc__ or ""
        reparsed = minidom.parseString(ET.tostring(manifest))
        return reparsed.toprettyxml(indent="  ")

    def CallTool(self, toolName: str, *args, **kwargs):
        if toolName in self.tools:
            return self.tools[toolName](*args, **kwargs)
        raise ValueError(f"Tool '{toolName}' not found.")

    def __str__(self):
        return self.Manifest()
    
    def __repr__(self):
        return self.Manifest()

class ProxyAgent(AgentBase):
    def __init__(self, name, role, manifest:str, tools: List[Callable] = []):
        self.name = name
        self.role = role
        self.manifest = str(manifest)
        self.tools = {}

        if isinstance(tools, list):
            for tool in tools:
                if not callable(tool):
                    raise ValueError("All tools must be callable.")
                else:
                    self.tools[tool.__name__] = tool
        else:
            self.tools = {}

    def Manifest(self):
        return self.manifest

class MicroAgent(AgentBase):
    def __init__(self, 
                 name:str, 
                 role:str, 
                 instructions:list[str], 
                 behaviors:list[str],
                 tools: List[Callable] = []):
        self.name = name
        self.role = role
        self.instructions = instructions
        self.behaviors = behaviors
        self.tools = {}

        self.XmlManifest = None

        if isinstance(tools, list):
            for tool in tools:
                if not callable(tool):
                    raise ValueError("All tools must be callable.")
                else:
                    self.tools[tool.__name__] = tool
        else:
            self.tools = {}

        self.__build()

    def Manifest(self):
        reparsed = minidom.parseString(ET.tostring(self.XmlManifest))
        return reparsed.toprettyxml(indent="  ")
    
    def __str__(self):
        return self.Manifest()
    
    def __repr__(self):
        return self.Manifest()
    
    def __build(self):
        if self.role is None:
            raise ValueError("Role must be defined for the agent.")
        
        if self.instructions is None:
            raise ValueError("At least one instruction must be provided for the agent.")

        if self.behaviors is None:
            raise ValueError("At least one behavior must be provided for the agent.")

        if not isinstance(self.instructions, list) or len(self.instructions) == 0:
            raise ValueError("Instructions must be a non-empty list of strings.")

        if not isinstance(self.behaviors, list) or len(self.behaviors) == 0:
            raise ValueError("Behaviors must be a non-empty list of strings.")

        manifest = ET.Element("AgentManifest")
        ET.SubElement(manifest, "role").text = self.role

        instructions = ET.SubElement(manifest, "instructions")
        for instruction in self.instructions:
            ET.SubElement(instructions, "instruction").text = instruction

        behaviors = ET.SubElement(manifest, "behaviors")
        for behavior in self.behaviors:
            ET.SubElement(behaviors, "behavior").text = behavior
        
        self.XmlManifest = manifest

class AdvancedAgent(MicroAgent):
    def __init__(self, 
                 name:str, 
                 role:str,
                 persona:str,
                 context:str,
                 instructions:list[str], 
                 behaviors:list[str],
                 outputFormat:str,
                 outputSchema:str,
                 tools: List[Callable] = []):
        super().__init__(name, role, instructions, behaviors, tools)
        self.persona = persona
        self.context = context
        self.outputFormat = outputFormat
        self.outputSchema = outputSchema
        self.__build_advanced()

    def __build_advanced(self):
        if self.persona is None:
            raise ValueError("Persona must be defined for the advanced agent.")

        if self.context is None:
            raise ValueError("Context must be defined for the advanced agent.")

        if self.outputFormat is None:
            raise ValueError("Output format must be defined for the advanced agent.")

        if self.outputSchema is None:
            raise ValueError("Output schema must be defined for the advanced agent.")

        manifest = ET.Element("AdvancedAgentManifest")
        ET.SubElement(manifest, "role").text = self.role
        ET.SubElement(manifest, "persona").text = self.persona
        ET.SubElement(manifest, "context").text = self.context
        ET.SubElement(manifest, "outputFormat").text = self.outputFormat
        ET.SubElement(manifest, "outputSchema").text = self.outputSchema

        instructions = ET.SubElement(manifest, "instructions")
        for instruction in self.instructions:
            ET.SubElement(instructions, "instruction").text = instruction

        behaviors = ET.SubElement(manifest, "behaviors")
        for behavior in self.behaviors:
            ET.SubElement(behaviors, "behavior").text = behavior

        self.XmlManifest = manifest