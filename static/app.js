// Show a "working" overlay while the server reads sheets / fits the model.
document.querySelectorAll("form[data-busy]").forEach(function (form) {
  form.addEventListener("submit", function () {
    document.getElementById("busyMsg").textContent = form.dataset.busy;
    document.getElementById("busy").hidden = false;
  });
});
// Dim a row when its status is switched to "dropped".
document.addEventListener("change", function (e) {
  if (e.target.classList.contains("st")) {
    e.target.closest("tr").classList.toggle("dropped", e.target.value === "dropped");
  }
});
// Click a day number to tick / untick that whole column for active rows.
document.querySelectorAll("table.grid thead th").forEach(function (th, idx) {
  if (idx < 4) return;
  th.title = "Click to tick / untick this day for all active students";
  th.style.cursor = "pointer";
  th.addEventListener("click", function () {
    var rows = Array.from(th.closest("table").tBodies[0].rows).filter(function (r) { return !r.classList.contains("dropped"); });
    var boxes = rows.map(function (r) { return r.cells[idx].querySelector("input"); });
    var all = boxes.every(function (b) { return b.checked; });
    boxes.forEach(function (b) { b.checked = !all; });
  });
});
