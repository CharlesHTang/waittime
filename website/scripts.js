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

    // TODO change this to match actual website
    document.getElementById("restaurant-name").textContent =
        data.restaurant;

    document.getElementById("wait-time").textContent =
        data.waitMinutes;

    document.getElementById("queue-count").textContent =
        data.ordersInQueue;
}

loadData();