import os
import json
import openai
from dotenv import load_dotenv
import requests
import base64
from google.generativeai import GenerativeModel
from deepseek_ai import DeepSeekClient
from midjourney import Midjourney
import ideogram

load_dotenv()  # 加载 .env 文件

openai.api_key = os.getenv("OPENAI_API_KEY")  # 设置 OpenAI API Key
gemini_api_key = os.getenv("GEMINI_API_KEY")  # 设置 Gemini API Key
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")  # 设置 DeepSeek API Key
midjourney_api_key = os.getenv("MIDJOURNEY_API_KEY") # 设置 Midjourney API Key
ideogram_api_key = os.getenv("IDEOGRAM_API_KEY") # 设置 Ideogram API Key

# 1. 数据读取
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

# 2. LLM Provider 接口
class LLMProvider:
    def call_api(self, prompt):
        raise NotImplementedError

# 3. OpenAI Provider
class OpenAIProvider(LLMProvider):
  def __init__(self, api_key):
    self.api_key = api_key
    openai.api_key = self.api_key

  def call_api(self, prompt):
    response = openai.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": "你是一个智能助手，帮助用户快速进行 Google Design Sprint"},
          {"role": "user", "content": prompt},
      ],
      temperature=0.1
    )
    return response.choices[0].message.content

# 4. Google Gemini Provider
class GeminiProvider(LLMProvider):
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = GenerativeModel('gemini-pro', api_key=self.api_key)

    def call_api(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text

# 5. DeepSeek AI Provider
class DeepSeekProvider(LLMProvider):
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = DeepSeekClient(self.api_key)

    def call_api(self, prompt):
        response = self.client.chat.completions.create(
             model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个智能助手，帮助用户快速进行 Google Design Sprint"},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content

# 6. 调用 LLM API
def call_llm(prompt, provider="openai"):
  if provider == "openai":
    llm_provider = OpenAIProvider(openai.api_key)
  elif provider == "gemini":
    llm_provider = GeminiProvider(gemini_api_key)
  elif provider == "deepseek":
        llm_provider = DeepSeekProvider(deepseek_api_key)
  else:
        raise ValueError(f"Invalid provider: {provider}")
  return llm_provider.call_api(prompt)

# 7. 输出结果存储
def save_output(output, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        if file_path.endswith(".json"):
            json.dump(json.loads(output), f, ensure_ascii=False, indent=2)  # ensure_ascii=False 用于支持中文
        else:
            f.write(output)

# 8. Image Provider 接口
class ImageProvider:
    def generate_image(self, prompt, output_path):
        raise NotImplementedError

# 9. Google ImageFX Provider
class GoogleImageFXProvider(ImageProvider):
    def generate_image(self, prompt, output_path):
        url = "https://labs.google/fx/tools/image-fx?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content="
        try:
            response = requests.get(url)
            response.raise_for_status()
            # 提取网页内容并分析，找到相应的图像API
            # 这部分需要对 google imagefx 进行爬虫并找到相应的api，相对复杂，此处不进行实现， 仅作为占位符
            print(f"正在使用 Google ImageFX 生成图片： {prompt}")
            # 根据Prompt和网页内容，调用Google ImageFX API生成图像，并保存到output_path
            # 由于无法直接调用其API，所以此处直接返回一个占位符的图像url
            image_url = "https://via.placeholder.com/400x300"
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(image_response.content)
            print(f"图片保存到: {output_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error when accessing the webpage: {e}")
            return None
        return output_path

# 10. Midjourney Provider
class MidjourneyProvider(ImageProvider):
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = Midjourney(api_key=self.api_key)

    def generate_image(self, prompt, output_path):
       try:
            print(f"正在使用 Midjourney 生成图片： {prompt}")
            result = self.client.imagine(prompt)
            if "image_url" in result:
                image_url = result['image_url']
                image_response = requests.get(image_url)
                image_response.raise_for_status()
                with open(output_path, "wb") as f:
                    f.write(image_response.content)
                print(f"图片保存到: {output_path}")
            else:
                 print(f"Midjourney 生成图片失败.")
                 return None
       except Exception as e:
            print(f"Error when accessing the Midjourney API : {e}")
            return None
       return output_path

# 11. Ideogram Provider
class IdeogramProvider(ImageProvider):
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = ideogram.Client(api_key = self.api_key)
    def generate_image(self, prompt, output_path):
        try:
            print(f"正在使用 Ideogram 生成图片： {prompt}")
            result = self.client.generate(prompt)
            if "image_url" in result:
               image_url = result['image_url']
               image_response = requests.get(image_url)
               image_response.raise_for_status()
               with open(output_path, "wb") as f:
                   f.write(image_response.content)
               print(f"图片保存到: {output_path}")
            else:
                 print(f"Ideogram 生成图片失败")
                 return None
        except Exception as e:
            print(f"Error when accessing the Ideogram API : {e}")
            return None
        return output_path
# 12. 调用 Image API
def generate_image(prompt, output_path, provider="google_imagefx"):
  if provider == "google_imagefx":
     image_provider = GoogleImageFXProvider()
  elif provider == "midjourney":
      image_provider = MidjourneyProvider(midjourney_api_key)
  elif provider == "ideogram":
      image_provider = IdeogramProvider(ideogram_api_key)
  else:
    raise ValueError(f"Invalid image provider: {provider}")
  return image_provider.generate_image(prompt, output_path)

# 13. Image Asset Extraction function
def extract_ui_elements(image_path):
    with open(image_path, 'rb') as file:
      image_data = file.read()
    image_b64 =  requests.compat.quote(base64.b64encode(image_data).decode("utf-8"))
    # image_b64 = base64.b64encode(image_data).decode("utf-8")
    asset_prompt = f"""Analyze the provided image and meticulously describe each UI element present, including all image-based assets (icons, backgrounds, illustrations). For each element, provide the following information:

         -   **type:** (e.g., button, text field, image, icon, background, etc.)
         -   **name:** (a unique name for each element)
         -   **bounding_box:** (y_min, x_min, y_max, x_max) in pixel values
         -   **description:** A detailed textual description of the element, including:
               - shape (e.g., square, circle, rounded rectangle, curved, lines, etc)
               - colors, including gradients if applicable (e.g., background color and foreground color)
               - textures or patterns (if any)
               -  whether it's a flat design or has any 3D effects like glowing, or shadow, etc.
              -  for icons, describe the illustration they contain
              -  for text, describe the font and if there is any style like boldness or italic
              - for background, describe the colors and gradients
         - **style**: A short description of overall visual style (e.g., cartoon, realistic, flat, neon, etc)
    Format your response as a JSON array, where each item represents a UI element and its properties. For images that have gradients, please include all the gradient colors. Be as comprehensive as possible.
          
          
        Here is the image with base64 encoding:
         {image_b64}"""
    asset_output = call_llm(asset_prompt)
    return asset_output

# 14. AI-Assisted Development Guidelines (Markdown)
ai_dev_guidelines = """
# AI-Assisted Development Guidelines

**1. Introduction:**
*   This document outlines a structured approach to AI-assisted development, emphasizing the importance of precision, especially with visual components. These guidelines are designed to help developers leverage AI tools like Cursor and Claude efficiently, minimize token usage, reduce errors, and faithfully reproduce layouts and designs from visual inputs, paying special attention to the position and dimensions of visual assets.

**2. Project Setup:**
*   A `fileNames.md` should list all files and directories with a one-line description of each component's purpose and function. This provides context for AI.
*   The `documentation` folder should contain: `prd.md` (Functional Requirements Document), `app-flow.md` (App Flow Diagram), `backend-structure.md` (Backend Architecture Overview), `frontend-guidelines.md` (Frontend Development Standards), `tech-stack.md` (Technology Stack Details), `file-structure.md` (Project File Structure), and `design-system.md`. (briefly mention what they should outline).
 * `design-system.md` file describes the core design components, such as fonts, colors, and spacing that are used throughout the app. Also include details on how to use placeholders to use in dev environment.
 *  Use `placeholders.shopconna.com` to generate placeholder images for development.

**3. Claude as "Software Architect":**
*   Set up a dedicated Claude project for refining prompts.
*   Add the following to Claude's knowledge base: `fileNames.md`, Full FRD, Component-specific FRDs, Cursor/bolt.new documentation, and `design-system.md`.

**4. Structured Prompting Flow:**
*   Use a two-step prompting flow: system prompt (context) and execution prompt (task).
*  System prompt should set context for Claude, including visual design considerations
*  Execution prompts should ask Claude to analyze problems, identify impacted files (using `fileNames.md`), and suggest approaches.
*   Example of execution prompt: "We need to add a login form based on the screenshot provided.  Refer to `src/components/login.jsx` from `fileNames.md` and  `design-system.md` from documentation folder, Please first provide analysis for the layout, text, image, font size, color scheme, and also the *bounding box, position (x,y coordinates), height, and width* for each visual element. After that, suggest an efficient approach for `bolt.new` to modify that file with the layout and elements described in the image, while using placeholders from `placeholders.shopconna.com` for the image assets, *maintaining the described position and dimension*."

**5. Cursor Prompting Techniques:**
*   **"Fix Errors"**: "Analyze this error. Identify its cause and create a step-by-step plan to resolve the issue, considering visual elements, *including their position and dimensions*."
*   **"New Feature"**: "Read the description in `@<document-name>` and `design-system.md` and create a plan for the implementation. Before proceeding, write an implementation plan. Explain what you are going to change before doing it, and pay special attention to the design of the components, *ensuring proper placement and dimensions of visual elements*."
    *   **Response Structure**: Provide updates and context, including visual component status. E.g., "The header is now aligned and using the correct font from `design-system.md` with correct size, and position. Now we need a login button with similar color. Check @login-doc and explain your approach, using `placeholders.shopconna.com` for the image, ensuring all assets are correctly positioned and sized".

**6. Progress Tracking:**
*   Use `progress.md` with this prompt: "At the end of each completed step, log your work in `@progress.md`. What features were implemented, what errors occurred, how were they fixed, and how were layout/design issues addressed, *including proper placement and dimensions of elements*? Answer these questions sequentially and do not miss information."
*   Use `project-status.md` with this prompt: "At the end of each session, log your work in `@project-status.md`. Review the `@progress.md` file to summarize all work and describe what was accomplished this session, including design implementations, *highlighting layout accuracy and proper asset dimensions*. Create a detailed report for the next working session, so there is a complete overview for the next session."

**7. Cursor Agent Hack:**
*  Prevent unnecessary changes: "Read @(document name) and `design-system.md` to determine the scope of the function and design. Using chain of thought logic, create a step-by-step implementation plan. Outline each part of the functionality with design specifics, *including precise positions, and dimension*, and their associated implementation steps.  Break those sections into detailed numbered steps. This will give a plan that you can approve and ensure that the actions conform with the requirements and designs."

**8. `.bolt/ignore` Optimization:**
*   Minimize context for LLM by using `.bolt/ignore` to exclude large image assets or design files.

**9. Handling Image Assets and Placeholders:**
*  Use placeholder images during the development.
*    Use `placeholders.shopconna.com` API to generate placeholder URLs with specific parameters like `width`, `height`, `text`, `bgColor`, `textColor`, and `fontFamily` as needed. E.g.,  Use  `https://placeholders.shopconna.com/?width=150&height=100&text=Placeholder` to create placeholder of 150 width and 100 height with text of "Placeholder". Adjust URL parameters to match desired style.  In production build, those placeholders should be replaced with actual prod assets, *ensuring they maintain the position and dimensions as defined in the designs*.

**10. Conclusion**
*   This approach enables developers to work efficiently while maintaining design fidelity by managing visual components and layout requirements, *emphasizing the importance of pixel-perfect implementation through accurate position and dimension handling*, while ensuring consistency.
"""

# 15. 执行 Google Design Sprint 的各个阶段
def run_design_sprint():
    print("----- Google Design Sprint 开始 -----")
    # --- Day 1: Map ---
    print("----- Day 1: Map -----")
    # (1) 从应用商店收集信息
    app_descriptions = read_data("data/app_store/app_descriptions.txt")
    user_reviews = read_data("data/app_store/user_reviews.txt")

    # Prompt for Application Description Analysis
    app_desc_prompt = f"""你是一个专业的应用分析师。请分析以下应用描述，提取出应用的核心功能、主要特点和目标用户群体。
        应用描述:
        {" ".join(app_descriptions)}

        请按照以下格式输出结果：
        - 核心功能: ...
        - 主要特点: ...
        - 目标用户群体: ...
        """
    app_desc_output = call_llm(app_desc_prompt)
    save_output(app_desc_output, "output/day1/app_descriptions_analysis.md")

   # Prompt for User Review Analysis
    review_prompt = f"""你是一个用户体验分析师。请分析以下用户评论，提取出用户提到的主要痛点、优点和改进建议。
      用户评论:
      {" ".join(user_reviews)}

      请按照以下格式输出结果：
      - 用户痛点: ... (如果没有痛点，则标注“无”)
      - 用户优点: ... (如果没有优点，则标注“无”)
      - 改进建议: ... (如果没有建议，则标注“无”)
      """
    review_output = call_llm(review_prompt)
    save_output(review_output, "output/day1/user_reviews_analysis.md")


    # (2) 从竞品网站收集信息
    competitor_files = [f for f in os.listdir("data/competitors/") if f.endswith((".html", ".md"))]
    for file in competitor_files:
        html_content = read_data(os.path.join("data/competitors/", file))

        # Prompt for HTML Analysis
        comp_prompt = f"""你是一个网页分析师。请分析以下 HTML 代码（或 Markdown 内容），提取出页面的主要结构和重要元素，例如导航栏、搜索框、主要内容区域、按钮、链接等。

            HTML 代码（或 Markdown）：
            {" ".join(html_content)}

            请按照以下格式输出结果：
            - 页面结构：... (例如，包含 header, main, footer 等)
            - 主要元素：... (例如，导航栏链接、搜索框、按钮文本、图片描述等)
            """
        comp_output = call_llm(comp_prompt)
        save_output(comp_output, f"output/day1/competitor_analysis_{os.path.splitext(file)[0]}.md")

        # Prompt for Functional Analysis
        comp_func_prompt = f"""你是一个功能分析师。请分析以下 HTML 代码（或 Markdown 内容），识别页面上的主要功能模块，例如表情符号搜索模块、表情符号编辑模块、表情符号组合模块等。

            HTML 代码（或 Markdown）：
            {" ".join(html_content)}

            请按照以下格式输出结果：
            - 功能模块：... (列出每个模块，并简要描述其功能)
            """
        comp_func_output = call_llm(comp_func_prompt)
        save_output(comp_func_output, f"output/day1/competitor_function_{os.path.splitext(file)[0]}.md")

         # Prompt for Content Analysis
        comp_content_prompt = f"""你是一个网页内容分析师。请分析以下 HTML 代码（或 Markdown 内容），提取页面上的关键文本内容（例如按钮标签、说明文本等）和交互元素（例如输入框、按钮、下拉列表等）。

            HTML 代码（或 Markdown）：
            {" ".join(html_content)}

            请按照以下格式输出结果：
            - 文本内容：... (列出关键的文本内容)
            - 交互元素：... (列出主要的交互元素及其类型)
            """
        comp_content_output = call_llm(comp_content_prompt)
        save_output(comp_content_output, f"output/day1/competitor_content_{os.path.splitext(file)[0]}.md")


    # (3) 从 Subreddit 收集信息
    subreddit_posts = read_data("data/subreddit/subreddit_posts.txt")

    # Prompt for Subreddit Analysis
    subreddit_prompt = f"""你是一个用户需求分析师。请分析以下 Subreddit 帖子/评论，提取用户表达的主要需求、痛点和问题。
        Subreddit 帖子/评论：
        {" ".join(subreddit_posts)}

        请按照以下格式输出结果：
        - 用户需求：... (列出用户表达的具体需求)
        - 用户痛点：... (列出用户遇到的具体问题和不便之处)
        """
    subreddit_output = call_llm(subreddit_prompt)
    save_output(subreddit_output, "output/day1/subreddit_analysis.md")

    subreddit_theme_prompt = f"""你是一个主题分类器。请分析以下 Subreddit 帖子/评论，并将其归类到以下预设主题中：
    {["功能", "用户体验", "bug", "想法", "竞品比较", "其他"]}

    Subreddit 帖子/评论：
      {" ".join(subreddit_posts)}

      请按照以下格式输出结果：
        - 主题分类：... (从预设主题中选择一个或多个)
      """
    subreddit_theme_output = call_llm(subreddit_theme_prompt)
    save_output(subreddit_theme_output, "output/day1/subreddit_theme_analysis.md")

    subreddit_sentiment_prompt = f"""你是一个情感分析师。请分析以下 Subreddit 帖子/评论，判断用户的情感倾向是积极、消极还是中性。
        Subreddit 帖子/评论：
        {" ".join(subreddit_posts)}
        请按照以下格式输出结果：
        - 情感倾向: ... (积极/消极/中性)
        """
    subreddit_sentiment_output = call_llm(subreddit_sentiment_prompt)
    save_output(subreddit_sentiment_output, "output/day1/subreddit_sentiment_analysis.md")

    # --- Day 2: Sketch ---
    print("----- Day 2: Sketch -----")
    # (1) 基于第一天收集的信息进行头脑风暴
    sketch_prompt = f"""你是一个创意解决方案师。请基于以下用户痛点，提出至少 3 个不同的解决方案，每个解决方案侧重解决一个或多个痛点。
        用户痛点：
        {review_output}

        请按照以下格式输出：
        - 解决方案 1：... (描述解决方案，侧重解决哪些痛点)
        - 解决方案 2：... (描述解决方案，侧重解决哪些痛点)
        - 解决方案 3：... (描述解决方案，侧重解决哪些痛点)
        """
    sketch_output = call_llm(sketch_prompt)
    save_output(sketch_output, "output/day2/sketch_ideas.md")

    tech_sketch_prompt = f"""你是一个技术创新师。请从技术角度出发，提出至少 3 个创新的解决方案，例如利用 AI、大数据等技术来改善工具的体验。

        请按照以下格式输出：
        - 技术方案 1：... (描述技术方案，侧重如何利用技术解决问题)
        - 技术方案 2：... (描述技术方案，侧重如何利用技术解决问题)
        - 技术方案 3：... (描述技术方案，侧重如何利用技术解决问题)
         """
    tech_sketch_output = call_llm(tech_sketch_prompt)
    save_output(tech_sketch_output, "output/day2/tech_sketch_ideas.md")

    ux_sketch_prompt = f"""你是一个用户体验专家。请从用户体验角度出发，提出至少 3 个解决方案，例如改善界面布局、操作流程等，让工具更易用、更便捷。

        请按照以下格式输出：
        - 用户体验方案 1：... (描述用户体验方案，侧重如何改善用户体验)
        - 用户体验方案 2：... (描述用户体验方案，侧重如何改善用户体验)
        - 用户体验方案 3：... (描述用户体验方案，侧重如何改善用户体验)
         """
    ux_sketch_output = call_llm(ux_sketch_prompt)
    save_output(ux_sketch_output, "output/day2/ux_sketch_ideas.md")

    crazy_sketch_prompt = f"""你是一个疯狂创意家。请尽可能提出一些“疯狂”的、不寻常的、甚至听起来不太可行的想法，来解决工具的问题，不要考虑技术和成本的限制。

        请按照以下格式输出：
        - 疯狂想法 1：... (描述疯狂的想法，鼓励发散思维)
        - 疯狂想法 2：... (描述疯狂的想法，鼓励发散思维)
        - 疯狂想法 3：... (描述疯狂的想法，鼓励发散思维)
    """
    crazy_sketch_output = call_llm(crazy_sketch_prompt)
    save_output(crazy_sketch_output, "output/day2/crazy_sketch_ideas.md")

    cross_sketch_prompt = f"""你是一个跨界思考者。请结合其他领域（例如游戏、社交、教育等）的创意，提出至少 3 个解决工具问题的方案。

       请按照以下格式输出：
        - 跨界方案 1：... (描述跨界方案，如何借鉴其他领域的创意)
        - 跨界方案 2：... (描述跨界方案，如何借鉴其他领域的创意)
        - 跨界方案 3：... (描述跨界方案，如何借鉴其他领域的创意)
        """
    cross_sketch_output = call_llm(cross_sketch_prompt)
    save_output(cross_sketch_output, "output/day2/cross_sketch_ideas.md")

    # --- Day 3: Decide ---
    print("----- Day 3: Decide -----")
    # (1) 快速批判（此处可以根据具体情况选择需要批判的）
    critique_prompt = f"""你是一个方案评估师。请针对以下选定的解决方案，列出至少 3 个优点，例如易用性、创新性、满足用户需求等。

        选定的解决方案描述：
        {sketch_output}

        请按照以下格式输出：
        - 方案优点：
          - 优点 1：...
          - 优点 2：...
          - 优点 3：...
        """
    critique_output = call_llm(critique_prompt)
    save_output(critique_output, "output/day3/solution_critique_advantages.md")

    critique_prompt_disadvantages = f"""你是一个方案评估师。请针对以下选定的解决方案，列出至少 3 个缺点或潜在的风险，例如技术难度、用户接受度、实现成本等。

        选定的解决方案描述：
        {sketch_output}
        请按照以下格式输出：
        - 方案缺点：
          - 缺点 1：...
          - 缺点 2：...
          - 缺点 3：...
        """
    critique_output_disadvantages = call_llm(critique_prompt_disadvantages)
    save_output(critique_output_disadvantages, "output/day3/solution_critique_disadvantages.md")

    feasibility_prompt = f"""你是一个方案可行性分析师。请基于以下选定的解决方案和我们目前的技术能力，分析该方案的可行性，并评估它在短期内实现的可能性。

       选定的解决方案描述：
       {sketch_output}
       目前的技术能力：
       我们有HTML/CSS/JS, React, node.js开发经验

       请按照以下格式输出：
       - 可行性分析：
         - 技术可行性：...
         - 短期实现可能性：...
       """
    feasibility_output = call_llm(feasibility_prompt)
    save_output(feasibility_output, "output/day3/feasibility.md")


    # (2) 故事板
    storyboard_prompt = f"""你是一个用户故事构建师。请基于以下选定的解决方案和用户角色，构建一个用户故事，描述用户如何使用该方案解决问题。

        选定的解决方案：
        {sketch_output}

        用户角色：
        {review_output}
        请按照以下格式输出：
        - 用户故事：
          - 场景：... (描述用户的使用场景)
          - 触发：... (描述用户触发该方案的动作)
          - 行为：... (描述用户使用该方案的具体行为)
          - 结果：... (描述用户使用该方案的结果)
        """
    storyboard_output = call_llm(storyboard_prompt)
    save_output(storyboard_output, "output/day3/storyboard.md")

    user_scene_prompt = f"""你是一个用户场景细化师。请基于以下用户故事，细化用户场景，描述用户在每个步骤的具体动作、感受和想法。
        用户故事：
        {storyboard_output}

        请按照以下格式输出：
        - 场景细节：
          - 步骤 1：... (描述用户在第一步的动作、感受和想法)
          - 步骤 2：... (描述用户在第二步的动作、感受和想法)
          - 步骤 3：... (描述用户在第三步的动作、感受和想法)
          - ... (更多步骤)
        """
    user_scene_output = call_llm(user_scene_prompt)
    save_output(user_scene_output, "output/day3/user_scene_detail.md")

    visual_prompt = f"""你是一个故事板视觉描述师。请基于以下细化的用户场景，描述故事板的视觉元素，例如需要绘制的界面、用户操作、交互流程等。
        细化的用户场景：
        {user_scene_output}

        请按照以下格式输出：
          - 视觉元素描述：
            - 场景 1：... (描述场景1的视觉元素)
            - 场景 2：... (描述场景2的视觉元素)
            - 场景 3：... (描述场景3的视觉元素)
            - ... (更多场景)
            """
    visual_output = call_llm(visual_prompt)
    save_output(visual_output, "output/day3/storyboard_visual_elements.md")

    # --- Day 4: Prototype ---
    print("----- Day 4: Prototype -----")
    # (1) 技术选型
    tech_prompt = f"""你是一个技术选型专家。请基于以下产品类型，推荐适合该类型产品的原型开发技术框架，并说明推荐理由。
        产品类型：
        网页应用

        请按照以下格式输出：
        - 技术框架推荐：... (推荐的技术框架名称)
        - 推荐理由：... (说明推荐该技术框架的原因)
        """
    tech_output = call_llm(tech_prompt)
    save_output(tech_output, "output/day4/tech_stack.json")
    # (2) 落地页
    landing_prompt = f"""你是一个文案策划师。请基于以下产品特点，生成简洁有力的落地页标题和副标题。
        产品特点：
          {product_description}

        请按照以下格式输出：
        - 落地页标题：... (简短有吸引力的标题)
        - 落地页副标题：... (进一步解释标题的副标题)
        """
    landing_output = call_llm(landing_prompt)
    save_output(landing_output, "output/day4/landing_page_title.md")

    landing_selling_prompt = f"""你是一个营销文案撰写人。请基于以下产品特点和用户痛点，生成吸引用户的关键卖点文案，并突出产品优势。

        产品特点：
           {product_description}
        用户痛点：
          {review_output}
        请按照以下格式输出：
        - 关键卖点文案：... (突出产品优势，解决用户痛点的文案)
        """
    landing_selling_output = call_llm(landing_selling_prompt)
    save_output(landing_selling_output, "output/day4/landing_page_selling_points.md")

    landing_visual_prompt = f"""你是一个视觉设计师。请基于以下产品特点和目标用户，提供落地页的视觉元素建议，例如颜色搭配、图片风格、布局方式等。

        产品特点：
            {product_description}
        目标用户：
            社交媒体重度用户

        请按照以下格式输出：
        - 颜色建议：... (推荐的颜色搭配方案)
        - 图片风格建议：... (推荐的图片风格)
        - 布局建议：... (推荐的页面布局方式)
        """
    landing_visual_output = call_llm(landing_visual_prompt)
    save_output(landing_visual_output, "output/day4/landing_page_visuals.md")

    # (3) 功能设计
    design_prompt = f"""你是一个功能分析师。请基于以下故事板和用户角色，明确原型需要展示的核心功能，并按优先级排序。
        故事板：
         {storyboard_output}
        用户角色：
          {review_output}
        请按照以下格式输出：
        - 核心功能：... (列出核心功能，并按优先级排序)
        """
    design_output = call_llm(design_prompt)
    save_output(design_output, "output/day4/prototype_design.json")

    interaction_prompt = f"""你是一个交互设计师。请基于以下故事板，定义原型的交互流程，描述用户如何与原型进行交互，完成特定任务。
          故事板：
           {storyboard_output}
          请按照以下格式输出：
          - 交互流程：... (详细描述用户如何与原型进行交互)
         """
    interaction_output = call_llm(interaction_prompt    save_output(interaction_output, "output/day4/prototype_interaction.md")

    # (4) 界面设计
    ui_prompt = f"""你是一个界面布局设计师。请基于以下原型功能和交互流程，提供界面元素布局建议，例如导航栏、搜索框、按钮、内容区域的布局方式。
         原型功能：
        {design_output}
        交互流程：
         {interaction_output}

        请按照以下格式输出：
        - 界面布局建议：... (描述页面元素布局的建议)
        """
    ui_output = call_llm(ui_prompt)
    save_output(ui_output, "output/day4/ui_layout.md")


    ui_element_prompt = f"""你是一个界面元素设计师。请基于以下原型需求，提供常用界面元素的设计建议，例如搜索框的样式、按钮的形状、图标的选择等。

        原型需求：
          {design_output}
        请按照以下格式输出：
        - 界面元素设计建议：
          - 搜索框样式：... (描述搜索框样式)
          - 按钮样式：... (描述按钮的形状和颜色)
          - 图标建议：... (描述图标的选择)
        """
    ui_element_output = call_llm(ui_element_prompt)
    save_output(ui_element_output, "output/day4/ui_elements.md")

    # --- Day 5: Test (Design Generation and Development Plan) ---
    print("----- Day 5: Test -----")
    # (1) 使用第四天的UI信息生成设计稿
    design_prompt = f"""你是一个视觉描述师。请根据以下界面布局和元素建议，描述一个简洁，现代的落地页设计稿
         界面布局：
         {ui_output}
         界面元素：
        {ui_element_output}
        请按照以下格式输出，确保包含主要视觉元素，并简洁清晰：
        整体设计：...
       """
    design_output = call_llm(design_prompt)
    image_output_path = "output/day5/design_mockup.png"
    image_provider = "google_imagefx"  # 默认provider
    # 可以根据用户选择使用不同的图片provider，这里为了演示，先随机选择一个
    import random
    providers = ["google_imagefx", "midjourney", "ideogram"]
    image_provider = random.choice(providers)
    image_output = generate_image(design_output, image_output_path, provider=image_provider)
    if image_output:
         print(f"{image_provider} 生成设计稿成功。")
    else:
         print(f"{image_provider} 生成设计稿失败。 使用占位符图片")

    # (1.1) Extract UI elements and Assets from generated image
    ui_elements_output = extract_ui_elements(image_output_path)
    save_output(ui_elements_output, "output/day5/ui_elements_from_design.json")

    # (2) 生成开发计划
    dev_plan_prompt = f"""你是一个项目经理。请基于以下信息，生成一份详细的开发计划，包括技术选型、落地页开发、原型功能开发、测试计划等，并按优先级排序。 同时遵循"AI-Assisted Development Guidelines"中的所有要求。
    
        技术选型：
        {read_data("output/day4/tech_stack.json")}
        落地页内容：
        {read_data("output/day4/landing_page_title.md")}
        {read_data("output/day4/landing_page_selling_points.md")}
        {read_data("output/day4/landing_page_visuals.md")}
        原型设计：
          {read_data("output/day4/prototype_design.json")}
          {read_data("output/day4/prototype_interaction.md")}
        UI设计:
          {read_data("output/day4/ui_layout.md")}
          {read_data("output/day4/ui_elements.md")}
       设计稿的图形资产描述：
         {ui_elements_output}
        AI辅助开发指南：
        {ai_dev_guidelines}

        请按照以下格式输出：
         - 开发计划：
             - 技术选型：... (描述选择的技术栈)
             - 落地页开发：... (描述落地页开发任务和优先级， 包括使用设计稿中的图形assets)
             - 原型功能开发：... (描述核心功能开发任务和优先级)
             - UI设计：... (描述UI实现任务，包括使用设计稿中的图形assets)
             - 测试计划：... (描述测试计划和重点)
             - 后续步骤：... (后续的步骤)
        """
    dev_plan_output = call_llm(dev_plan_prompt)
    save_output(dev_plan_output, "output/day5/development_plan.md")
    print("----- Google Design Sprint 完成 -----")

# 产品描述 (通用)
product_description = "Describe your product here"

# 创建输出目录
os.makedirs("output/day1", exist_ok=True)
os.makedirs("output/day2", exist_ok=True)
os.makedirs("output/day3", exist_ok=True)
os.makedirs("output/day4", exist_ok=True)
os.makedirs("output/day5", exist_ok=True)

# 运行 Google Design Sprint
run_design_sprint()
