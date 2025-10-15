# Claude Notes

This file contains notes and documentation for Claude to reference in future sessions.

## GitHub README Layout Tips

When adding images to GitHub README files:

- **Avoid tables for layout**: GitHub's markdown renderer applies default table styling with visible borders that cannot be overridden with inline CSS styles
- **Use floated images instead**: Use `<img>` tags with `align="right"` or `align="left"` attributes wrapped in a `<div>` for better control
- **Example pattern**:
  ```html
  <div>
  <a href="https://example.com">
    <img src="image.png" alt="Description" width="250" align="right">
  </a>

  Your markdown content here flows naturally around the image...
  </div>
  ```

This approach provides clean layouts without visible borders and allows text to flow naturally around images.
