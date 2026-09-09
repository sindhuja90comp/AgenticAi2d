"""Create the small, reusable cartoon layer packs supported in Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from PIL import Image, ImageDraw

from ..models.phase2 import BirdFenceGeometry


class Scenario(StrEnum):
    BUTTERFLY_FLOWER = "butterfly_flower"
    BIRD_FENCE = "bird_fence"


@dataclass(frozen=True, slots=True)
class GeneratedAssetPack:
    scenario: Scenario
    background: Path
    subject: Path
    mouth_sheet: Path
    accent: Path | None = None
    parts: dict[str, Path] | None = None


class Phase1AssetGenerator:
    """Draw clean, deterministic line-art PNGs without an external image service."""

    WIDTH = 1920
    HEIGHT = 1080

    def __init__(self, root: Path | str = "assets/generated") -> None:
        self._root = Path(root)

    def generate(self, scenario: Scenario, *, geometry: BirdFenceGeometry | None = None) -> GeneratedAssetPack:
        directory = self._root / scenario.value
        directory.mkdir(parents=True, exist_ok=True)
        background = directory / "background.png"
        subject = directory / ("butterfly.png" if scenario is Scenario.BUTTERFLY_FLOWER else "bird.png")
        mouth_sheet = directory / "mouth_sheet.png"
        accent = directory / "flower.png" if scenario is Scenario.BUTTERFLY_FLOWER else None
        parts: dict[str, Path] = {}
        if scenario is Scenario.BUTTERFLY_FLOWER:
            self._butterfly_background(background)
            self._butterfly(subject)
            self._flower(accent)
        else:
            self._bird_background(background, geometry)
            self._bird(subject)
            parts = {
                "body": directory / "bird_body.png",
                "left_wing": directory / "bird_left_wing.png",
                "right_wing": directory / "bird_right_wing.png",
            }
            self._bird_parts(parts["body"], parts["left_wing"], parts["right_wing"])
        self._mouth_sheet(mouth_sheet)
        return GeneratedAssetPack(scenario, background, subject, mouth_sheet, accent, parts)

    def _canvas(self, color: str = "#dff4ff") -> tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("RGBA", (self.WIDTH, self.HEIGHT), color)
        return image, ImageDraw.Draw(image)

    def _butterfly_background(self, path: Path) -> None:
        image, draw = self._canvas("#dff5ff")
        draw.rectangle((0, 760, self.WIDTH, self.HEIGHT), fill="#a9d879")
        draw.ellipse((730, 470, 1190, 1120), fill="#5f9f3b", outline="#315d27", width=12)
        draw.line((960, 850, 960, 560), fill="#3d7a37", width=24)
        for x, y in ((820, 610), (1080, 650), (880, 710), (1040, 730)):
            draw.ellipse((x - 65, y - 35, x + 65, y + 35), fill="#f9e06c", outline="#6b5520", width=8)
        draw.ellipse((895, 500, 1025, 630), fill="#f5bd3f", outline="#6b5520", width=10)
        image.save(path)

    def _butterfly(self, path: Path) -> None:
        image = Image.new("RGBA", (700, 700), (0, 0, 0, 0)); draw = ImageDraw.Draw(image)
        draw.ellipse((90, 135, 340, 430), fill="#8f8be7", outline="#332c6d", width=16)
        draw.ellipse((360, 135, 610, 430), fill="#ee8bc8", outline="#71345b", width=16)
        draw.ellipse((275, 155, 425, 535), fill="#303b54", outline="#161d29", width=14)
        draw.line((325, 170, 250, 70), fill="#161d29", width=12); draw.line((375, 170, 450, 70), fill="#161d29", width=12)
        image.save(path)

    def _flower(self, path: Path) -> None:
        image = Image.new("RGBA", (600, 700), (0, 0, 0, 0)); draw = ImageDraw.Draw(image)
        draw.line((300, 680, 300, 290), fill="#3d7a37", width=28)
        for x, y in ((180, 300), (420, 300), (300, 190), (300, 410)):
            draw.ellipse((x - 110, y - 70, x + 110, y + 70), fill="#f9e06c", outline="#6b5520", width=12)
        draw.ellipse((230, 230, 370, 370), fill="#f5bd3f", outline="#6b5520", width=12)
        image.save(path)

    def _bird_background(self, path: Path, geometry: BirdFenceGeometry | None = None) -> None:
        image, draw = self._canvas("#ccecff")
        draw.rectangle((0, 760, self.WIDTH, self.HEIGHT), fill="#91c66c")
        if geometry is None:
            # Reproduce saved pre-geometry previews when resuming their approval.
            draw.rectangle((0, 690, self.WIDTH, 790), fill="#a56b3f", outline="#5d3620", width=14)
            for x in range(80, self.WIDTH, 250):
                draw.rectangle((x, 620, x + 55, 1030), fill="#8b5635", outline="#5d3620", width=12)
        else:
            # PIL includes both endpoints: subtract one at the exclusive bottom/right
            # edges so the rod and poles touch without sharing a row of pixels.
            for x in geometry.pole_x_positions:
                draw.rectangle((x, geometry.pole_top_y, x + geometry.pole_width - 1, geometry.pole_bottom_y - 1), fill="#8b5635", outline="#5d3620", width=12)
            draw.rectangle((geometry.rod_left_x, geometry.rod_top_y, geometry.rod_right_x - 1, geometry.pole_top_y - 1), fill="#a56b3f", outline="#5d3620", width=14)
        image.save(path)

    @classmethod
    def bird_foot_y_percent(cls) -> float:
        body = Image.new("RGBA", (620, 620), (0, 0, 0, 0))
        cls._draw_bird_body(body)
        return body.getchannel("A").getbbox()[3] / body.height * 100

    @classmethod
    def bird_landing_y_percent(cls, geometry: BirdFenceGeometry, output_width: int = 1920,
                               output_height: int = 1080) -> float:
        """Place the scaled visible foot edge on the rod, ignoring sprite padding.

        All parts use the same 620px canvas and center, so they share this baseline.
        Scale rounding matches LocalFfmpegRenderer's sprite preparation.
        """
        size = max(1, round(output_width * 0.35 * geometry.bird_scale))
        rod_top = geometry.rod_top_y * output_height / cls.HEIGHT
        return (rod_top - (cls.bird_foot_y_percent() / 100 - 0.5) * size) / output_height * 100

    def _bird(self, path: Path) -> None:
        image = Image.new("RGBA", (620, 620), (0, 0, 0, 0))
        self._draw_left_wing(image)
        self._draw_right_wing(image)
        self._draw_bird_body(image)
        image.save(path)

    def _bird_parts(self, body_path: Path, left_wing_path: Path, right_wing_path: Path) -> None:
        body = Image.new("RGBA", (620, 620), (0, 0, 0, 0))
        self._draw_bird_body(body)
        body.save(body_path)
        left_wing = Image.new("RGBA", (620, 620), (0, 0, 0, 0))
        self._draw_left_wing(left_wing)
        left_wing.save(left_wing_path)
        right_wing = Image.new("RGBA", (620, 620), (0, 0, 0, 0))
        self._draw_right_wing(right_wing)
        right_wing.save(right_wing_path)

    @staticmethod
    def _draw_bird_body(image: Image.Image) -> None:
        """Draw a readable side-view bird while keeping the multipart layer pivots stable."""
        draw = ImageDraw.Draw(image)
        outline = "#1d365f"
        # Tail and feet make the pose read as a perched bird rather than a floating oval.
        draw.polygon(((190, 300), (55, 365), (180, 405), (245, 365)), fill="#315f9c", outline=outline, width=12)
        draw.polygon(((210, 320), (75, 420), (225, 425), (280, 370)), fill="#487fc1", outline=outline, width=10)
        draw.line((292, 425, 292, 510), fill="#71452b", width=14)
        draw.line((382, 425, 382, 510), fill="#71452b", width=14)
        for x in (292, 382):
            draw.line((x, 505, x - 30, 520), fill="#71452b", width=9)
            draw.line((x, 505, x + 26, 520), fill="#71452b", width=9)
        draw.ellipse((150, 205, 455, 445), fill="#4f86cf", outline=outline, width=16)
        draw.ellipse((250, 270, 438, 435), fill="#d9ecff", outline=outline, width=9)
        draw.ellipse((335, 115, 505, 295), fill="#6098dc", outline=outline, width=16)
        draw.polygon(((378, 142), (412, 75), (435, 145)), fill="#3d70af", outline=outline, width=9)
        draw.ellipse((405, 160, 456, 211), fill="#f6fbff", outline=outline, width=7)
        draw.ellipse((424, 177, 445, 198), fill="#16233d")
        draw.polygon(((492, 210), (590, 254), (492, 290)), fill="#f3b648", outline="#805416", width=10)
        draw.arc((355, 210, 468, 275), start=15, end=145, fill="#274e85", width=8)

    @staticmethod
    def _draw_left_wing(image: Image.Image) -> None:
        """The root stays at x=225 (36.3%) for the existing shoulder pivot contract."""
        draw = ImageDraw.Draw(image)
        outline = "#1d365f"
        draw.polygon(
            ((225, 250), (120, 225), (42, 278), (82, 325), (35, 365), (115, 410), (225, 390), (274, 330)),
            fill="#2d669f", outline=outline, width=13,
        )
        for offset in (0, 35, 70):
            draw.line((220, 278 + offset // 3, 82 + offset, 326 + offset), fill="#a9d1f2", width=7)

    @staticmethod
    def _draw_right_wing(image: Image.Image) -> None:
        """The root stays at x=395 (63.7%) for the existing shoulder pivot contract."""
        draw = ImageDraw.Draw(image)
        outline = "#1d365f"
        draw.polygon(
            ((395, 250), (488, 220), (575, 270), (540, 325), (590, 365), (510, 410), (395, 390), (346, 330)),
            fill="#3776b7", outline=outline, width=13,
        )
        for offset in (0, 35, 70):
            draw.line((400, 278 + offset // 3, 538 - offset, 326 + offset), fill="#b7dcfa", width=7)

    def _mouth_sheet(self, path: Path) -> None:
        image = Image.new("RGBA", (180, 9 * 70), (0, 0, 0, 0)); draw = ImageDraw.Draw(image)
        for index in range(9):
            y = index * 70
            draw.ellipse((45, y + 22, 135, y + 48), fill="#3c1e27")
        image.save(path)
