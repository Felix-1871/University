import { defineStore } from 'pinia';
import { skills } from '../data/skills.json';
export const useSkillsStore = defineStore( "SkillsStore", {
  state: () => {
  return {
    skills : skills,
  };
    
  },
})
