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