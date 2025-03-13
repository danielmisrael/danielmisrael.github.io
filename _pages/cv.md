---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

<iframe src="/assets/pdf/your-cv.pdf" width="100%" height="800px"></iframe>

[Download my CV (PDF)](/assets/pdf/your-cv.pdf) ```

* **Adjust `width` and `height`:**
    * Modify the `width` and `height` attributes of the `<iframe>` tag to fit your desired display.

**3. PDF Embed using Google Docs Viewer**

This method uses Google Docs Viewer to display your PDF, which often provides better cross-browser compatibility.

* **Prepare your PDF:**
    * Upload your PDF to a publicly accessible location (e.g., your GitHub Pages repository, as in the previous examples).
* **Modify your `cv.md` file:**

```markdown
---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

<iframe src="https://docs.google.com/viewer?url=https://your-github-username.github.io/your-repo-name/assets/pdf/your-cv.pdf&embedded=true" width="100%" height="800px"></iframe>

[Download my CV (PDF)](/assets/pdf/CV [2_16_24].pdf)
