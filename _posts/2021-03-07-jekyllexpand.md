---
layout: post
title: "Jekyll Blog Collapsible Block 만들기"
date: "2021-03-07 18:28:01 +0900"
categories: programming
author: "Soo"
comments: true
toc: true
---

# 지킬블로그 텍스트 확장 블록 만들기

지킬에서 블로그를 쓰다보면 가끔 아주 긴 부연설명에 대한 텍스트를 넣고 싶거나, 긴 코드블록을 숨겨서 이쁘게 꾸미고 싶을 때가 있다. 그런데 내가 인터넷에서 나오는 expand 모듈들은 markdown 내에 html 태그를 길게 써야되서 불편했다. 그래서 기존에 인터넷에 있던 코드를 기반으로 새로 만들었다. 그렇게 썩 깔끔한 코드는 아니지만, 해당기능이 필요한 사람들에게 잘 사용됐으면 좋겠다.

---

# 세팅법

제일먼저 세팅해야할 것은 `_config.yml` 에서 마크다운이 `kramdown` 인지 확인 하는 것이다. 그 이유는 collapsible block을 만들기 위해서는 `<details>` 태그를 사용해야하는데,  `kramdown`이 `<details>`를 지원하는 것으로 알고 있다.

```yaml
# _config.yml
markdown: kramdown
```

[expand]summary:열어서 text-expander 코드 복사하기 👈 

[여기](https://gist.github.com/simonjisu/43c789bf44e9f8171be440b46f0948a5)에서 다운로드 하거나, 아래 코드를 복사한다.

```html
<!-- Author: https://github.com/simonjisu
Change `div.article-content` to your article container in jekyll blog
Put your file into `_include/text-expand.html`
-->
<script>
var elements = document.querySelectorAll('div.article-content')[0].childNodes; // 수정1 
var addContent = false;
var contentsToAdd = [];
var expandtags = null;
var detailText = null;
var detailsTag = null;
var summaryTag = null;
var detailsContent = null;
for (var i=elements.length - 1; i > -1; i--){
    el = elements[i]
    if (el.innerHTML == '[/expand]'){
        addContent = true
        detailsContent = document.createElement('div')
        detailsContent.className = 'collaspe-content'
        detailsContent.setAttribute('markdown', '1')
        el.parentNode.removeChild(el)
    } else if (el.innerHTML == '[expand]' || (el.nodeName == 'P' && el.innerHTML.includes('[expand]summary:'))) {
        addContent = false
        expandtags = el.innerHTML.split('summary:')
        if (expandtags.length == 1){
            detailText = 'Details'
        } else {
            detailText = expandtags[1]
        }
        detailsTag = document.createElement('details')
        detailsTag.className = 'collaspe-article'
        summaryTag = document.createElement('summary')
        summaryTag.appendChild(document.createTextNode(detailText))
        detailsTag.appendChild(summaryTag)
        for (var j=contentsToAdd.length - 1; j > -1; j--) {
            detailsContent.appendChild(contentsToAdd[j])
        }
        detailsTag.appendChild(detailsContent)
        el.parentNode.replaceChild(detailsTag, el)
        contentsToAdd = []
    } else {
        if (addContent) {
            contentsToAdd.push(el)
        }
    }
}
</script>
```

[/expand]

그 다음 `_include` 폴더내에 `text-expand.html` 파일을 만들고 다음 코드를 복사해서 붙여넣기 하자. 여기서 수정할 부분은 다음과 같다.

```javascript
var elements = document.querySelectorAll('div.article-content')[0].childNodes;
```

`div.article-content` 부분을 수정해야하는데 자신의 jekyll 구조를 파악해서 글의 내용이 어느 컨테이너에 있는지 확인해야한다. 자신의 블로그에서 마우스 오른쪽 버튼을 누르고 `검사`를 통해 구조를 파악하거나, `_layout`폴더의 파일들 중 `<body>` 태그 사이를 잘 살펴보면 된다. 내 블로그의 경우 구조가 다음과 같은데, `<div class="article-content">` 가 글에 해당하는 내용이다.

```html
<!DOCTYPE html>
<html>
  <head> ... </head>
  <body>
    <div class="page-content">
      <div class="container">
        <div class="three columns">
          <header> ... </header>
        </div>
        <div class="nine columns" style="z-index:100;">
          <div class="wrapper">
            <article class="post">
            <header class="post-header">
              <h1 class="post-title"> ... </h1>
            </header>
            <div class="article-content">
                content   <!-- 포스트의 내용이 담김 곳 -->
            </div>
            </article>
          </div>
        </div>
      </div>
      <footer> ... </footer>
    </div>
  </body>
</html>
```

그 다음 스텝으로 `_layout` 폴더에서 `</body>` 태그가 들어간 파일을 찾아, 이전에 다음과 같이 liquid 문법으로 아까 만든 `text-expand.html`을 포함시킨다. 

```html
{% raw %}{% include text-expand.html %}{% endraw %}
</body>
```

마지막으로 `_sass` 폴더의 `_layout.scss` 파일에 관련 css만 추가해주면 끝난다.

```scss
// _sass/_layout.scss 
.collaspe-article {
  padding-top: 10px;
  padding-bottom: 10px;
}
.collaspe-content{
  padding-top: 5px;
}
.collaspe-content:before {
  content: "";
  display: block;
  width: 100%;
  border-bottom: 1px solid #bcbcbc;
}
.collaspe-content:after {
  content: "";
  display: block;
  width: 100%;
  border-bottom: 1px solid #bcbcbc;
}
```

---

# 사용법

마크다운에서 다음과 같이 쓰면 된다. 주의할 점은 `[expand]`사이에 새 줄만 잘 띄어주면 된다. `[expand]`뒤에는 `summary:`를 붙여서 설명하고 싶은 내용을 적을 수 있다. 만약에 없으면 기본으로 `Details`가 들어간다.

예를 들면, 다음 코드는 아래처럼 바뀐다.

[expand]summary:원하는 블록 요약 쓰기

내용을 써주세요. [expand] 사이에 마크다운 문법이 가능합니다.

$$1 + 1 = 3$$

설명을 위해 코드블록을 일부러 띄워 썼습니다. 실제로 쓸때는 밑에 띄어쓴 칸을 지우세요!
    
```python
def add(a, b):
    return a + b
```

[/expand]

```markdown

[expand]summary:원하는 블록 요약 쓰기

내용을 써주세요. [expand] 사이에 마크다운 문법이 가능합니다.

$$1 + 1 = 3$$

설명을 위해 코드블록을 일부러 띄워 썼습니다. 실제로 쓸때는 밑에 띄어쓴 칸을 지우세요!
    
    ```python
    def add(a, b):
        return a + b
    ```

[/expand]

```

---

# 기본원리

짧게 설명하면, Markdown에서 Collapsible block의 문법은 다음과 같으며, 원하면 자신만의 코드로 커스텀해서 사용해볼 수 있다.

```html
<details>
<summary> 표기할것 </summary>
내용쓰기
</details>
```
