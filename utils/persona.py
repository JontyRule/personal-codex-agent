from __future__ import annotations

from pydantic import BaseModel
from typing import List, Dict


class Voice(BaseModel):
    tone: str
    sentence_length: str
    forbid: List[str]
    signature_phrases: List[str]


class Profile(BaseModel):
    name: str
    title: str
    summary: str
    voice: Voice
    values: List[str]
    strengths: List[str]
    growth_areas: List[str]
    culture: List[str]
    debugging_style: str

    def to_prompt_block(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Title: {self.title}\n"
            f"Summary: {self.summary}\n"
            f"Voice: tone={self.voice.tone}; sentence_length={self.voice.sentence_length}; "
            f"forbid={', '.join(self.voice.forbid)}; signatures={'; '.join(self.voice.signature_phrases)}\n"
            f"Values: {', '.join(self.values)}\n"
            f"Strengths: {', '.join(self.strengths)}\n"
            f"Growth Areas: {', '.join(self.growth_areas)}\n"
            f"Culture: {', '.join(self.culture)}\n"
            f"Debugging Style: {self.debugging_style}\n"
        )
