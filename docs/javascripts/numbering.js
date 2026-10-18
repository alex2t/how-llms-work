/* Number the sections (3.1) and subsections (3.1.1) of each chapter, in the page and in its table of contents.
   The chapter number comes from the page address (03-inference/ -> 3); pages without one are left as they are. */
document.addEventListener("DOMContentLoaded", function () {
  var match = window.location.pathname.match(/\/(\d\d)-[^/]+\/(index\.html)?$/);
  if (!match) return;
  var chapter = String(Number(match[1]));
  var section = 0;
  var subsection = 0;

  document.querySelectorAll(".md-content .md-typeset h2, .md-content .md-typeset h3").forEach(function (heading) {
    if (heading.tagName === "H2") {
      section += 1;
      subsection = 0;
    } else {
      subsection += 1;
    }
    var number = chapter + "." + section + (heading.tagName === "H3" ? "." + subsection : "");
    heading.insertAdjacentText("afterbegin", number + " ");
    document.querySelectorAll('.md-nav__link[href="#' + heading.id + '"] .md-ellipsis').forEach(function (label) {
      label.insertAdjacentText("afterbegin", number + " ");
    });
  });
});
