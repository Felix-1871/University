<template>
    <div class="MainContainer_base">
        <Navbar />
        <div class="MainContainer">
            <RouterView />
        </div>
    </div>
</template>

<style scoped>

.MainContainer {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    border: 2px solid #fff;
    border-radius: 10px;
    background-color: var(--ctp-mocha-crust);
}



</style>