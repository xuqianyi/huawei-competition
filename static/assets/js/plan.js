const main = document.querySelector('main');
let chapters = 5;
let numChapter = 5;
let htmlString = "";
let n = parseFloat(document.getElementById("days").textContent);

const totalChapters = 5;
let chapterList = Array.from({length: totalChapters}, (_, i) => i + 1); // [1, 2, 3, 4, 5]

let division = [];
let base = Math.floor(totalChapters / n);
let extra = totalChapters % n;

for (let i = 0; i < n; i++) {
    let currentGroupSize = base + (i < extra ? 1 : 0);
    division.push(chapterList.splice(0, currentGroupSize));
}
htmlString += `<ul class="plan-list">`
for (let i = 0; i < n; i++) {
    let chapterLinks = division[i].map(chapter => 
        `<a href="/learn#chapter${chapter}.html">Chapter ${chapter}</a>`
    );
    htmlString += `
    <li style="--accent-color:#41516C">
        <div class="date">Day ${i + 1}</div>
        <div class="title">Practice</div>
        <div class="descr">${chapterLinks.join(', ')}</div>
    </li>
    `;
}
htmlString += `</ul>`
main.innerHTML = htmlString; 
console.log(htmlString);


