document.addEventListener("DOMContentLoaded", function () {
    const toggle = document.querySelector(".menu-toggle");
    const nav = document.querySelector(".nav-links");
    const links = nav.querySelectorAll("a .header-links");

    function openMenu() {
        nav.classList.add("show");
        document.addEventListener("click", outsideClickListener);
    }

    function closeMenu() {
        nav.classList.remove("show");
        document.removeEventListener("click", outsideClickListener);
    }

    function outsideClickListener() {
        closeMenu();
    }

    toggle.addEventListener("click", (e) => {
        e.stopPropagation(); // prevent click from immediately triggering outsideClick
        if (nav.classList.contains("show")) {
            closeMenu();
        } else {
            openMenu();
        }
    });

    links.forEach(link => {
        link.addEventListener("click", () => {
            closeMenu();
        });
    });
});

async function loadData() {
    const response = await fetch("/store1data.json");
    const data = await response.json();

    const waitTimeElements = document.querySelectorAll(".wait-time");

    for (const element of waitTimeElements) {
        element.textContent = "Wait Time: " + Math.round(data.average_wait_seconds / 60) + " min";
    }

    const lineCountElements = document.querySelectorAll(".line-count");

    for (const element of lineCountElements) {
        element.textContent = "Line Length: " + data.active_people_in_line + " people";
    }
}

loadData();
setInterval(loadData, 30_000);