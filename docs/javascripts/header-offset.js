/* Record the header's height (it grows when the chapter tabs wrap onto several rows),
   so that tabs.css can keep anchor targets clear of it. */
function setHeaderHeight() {
  var height = document.querySelector(".md-header").offsetHeight;
  document.documentElement.style.setProperty("--header-height", height + 16 + "px");
}

document.addEventListener("DOMContentLoaded", setHeaderHeight);
window.addEventListener("resize", setHeaderHeight);
