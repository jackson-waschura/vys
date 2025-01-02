"""
This is a simple experiment to see how to use Manim to create visualizations.

It can be rendered with the following command:

```
uv run manim -pql simple_scene.py SimpleScene
```

The `-pql` flag is used to render the video in low quality. It means:

- `p`: render a preview
- `ql`: use the lowest quality setting
"""

from manim import *

class SimpleScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
