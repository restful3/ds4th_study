import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


RENDER_PATH = Path(__file__).parents[1] / "scripts" / "render.py"
SPEC = importlib.util.spec_from_file_location("localize_image_render", RENDER_PATH)
render_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(render_module)


class RenderEllipseTest(unittest.TestCase):
    def test_ellipse_supports_outline_and_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            output = root / "output.png"
            spec = root / "spec.json"
            Image.new("RGB", (40, 40), "white").save(source)
            spec.write_text(
                json.dumps(
                    {
                        "title": "ellipse outline",
                        "source": "source.png",
                        "output": "output.png",
                        "coordinate_space": [40, 40],
                        "operations": [
                            {
                                "type": "ellipse",
                                "box": [10, 10, 20, 20],
                                "fill": "#ffffff",
                                "outline": "#000000",
                                "width": 3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            render_module.render_spec(spec, {})
            result = Image.open(output).convert("RGB")
            self.assertEqual(result.getpixel((20, 10)), (0, 0, 0))
            self.assertEqual(result.getpixel((20, 20)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
