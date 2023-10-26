<template>
    
    <div class="Games">
        <div class="Games__list">
            <div class="Games__item" v-for="game in games" :key="game.dealID">
                <div class="Games__item__image">
                    <img :src="game.thumb" alt="Game Image" />
                </div>
                <div class="Games__item__title">
                    <h3>{{ game.title }}</h3>
                </div>
                <div class="Games__item__price">
                    <p>{{ game.normalPrice }}$</p>
                </div>
                <div class="Games__item__rating">
                    <p :style="{ color: `hsl(${game.steamRatingPercent * 1.2}, 100%, 50%)` }">Metacritic rating: {{ game.steamRatingPercent }}%</p>
                </div>
                <div class="Games__item__link">
                    <button @click="AddToCart(game)">Add to cart</button>
                </div>
            </div>
        </div>
    </div>
</template>

<script>

import { createClient } from '@supabase/supabase-js'


 const supabaseUrl = 'https://ttgymjmisrdktsonzgoa.supabase.co'
 const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR0Z3ltam1pc3Jka3Rzb256Z29hIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODYwODMxMDAsImV4cCI6MjAwMTY1OTEwMH0.AjFgT0mwVjlJm8H_lcajqZTXU7C72epYg4uLpYenosY'
 const supabase = createClient(supabaseUrl, supabaseKey)

export default {
  data() {
    return {
      games: []
    }
    
  },
  props: {
      game: Object
    },
  methods: {
    async AddToCart(game) {
  // Insert the game into the cart table
  const { data, error } = await supabase
    .from('cart')
    .insert([{ title: game.title, price: game.normalPrice }])

  if (error) {
    // Handle the error
    console.log(error)
  } else {
    // Show a success message
    alert('Game added to cart!')
    console.log(data)
  }
}
  },
  async created() {
    const response = await fetch('https://www.cheapshark.com/api/1.0/deals?storeID=1&upperPrice=15')
    const data = await response.json()
    if (response) {
      console.log(data)
      this.games = data
    } else {
      console.log('error')
    }
  }
}


</script>

<style scoped>

.Games {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
    height: 100%;
    overflow: scroll;
}

.Games__list {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

.Games__item {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 200px;
    margin: 20px;
    padding: 20px;
    border: 3px solid var(--ctp-mocha-text);
    border-radius: 10px;
    box-shadow: 0 0 10px var(--ctp-mocha-text);
}

.Games__item__image {
    width: 100%;
    

}



.Games__item__image img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.Games__item__title {
    margin: 10px 0;
}

.Games__item__price {
    margin: 10px 0;
}

.Games__item__link {
    margin: 10px 0;
    width: 100%;
    text-align: center;
    height: 100%;
    background-color: var(--ctp-frappe-base);
    border-radius: 10px;
    padding: 10px;
    
}

.Games__item__link button {
    width: 100%;
    height: 100%;
    background-color: var(--ctp-frappe-base);
    border: none;
    border-radius: 10px;
    padding: 10px;
    color: var(--ctp-mocha-text);
    font-weight: bold;
    cursor: pointer;
}

.Games__item__link button:hover {
    background-color: var(--ctp-mocha-text);
    color: var(--ctp-frappe-base);
}








</style>

