<template>
    <div class="Navbar">
        <div class="Navbar__logo">
            <img src="~/assets/logo.png" alt="Logo" />
        </div>
        <div class="Navbar__links">
            <NuxtLink to="/">Home</NuxtLink>
            <NuxtLink to="/about">About</NuxtLink>
            <NuxtLink to="/games">Games</NuxtLink>
            <NuxtLink to="/contact">Contact</NuxtLink>
            <NuxtLink to="/login">Login</NuxtLink>
            <NuxtLink to="/register">Register</NuxtLink>
            

        </div>
    </div>
</template>

<style scoped>

.Navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100vh;
    margin: 0 auto;
    padding: 0 20px;
    height: 100px;
    border: 1px solid #fff;
    border-radius: 10px;

    background-color: var(--ctp-mocha-mantle);
}

.Navbar__logo {
    width: 100px;
}

.Navbar__links {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 50%;
}

.Navbar__links a {
    text-decoration: none;
    color: var(--ctp-mocha-blue);

}

.Navbar__links a:hover {
    color: var(--ctp-mocha-text);

}



</style>