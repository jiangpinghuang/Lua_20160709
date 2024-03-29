---- Chap 28
--
--int l_map (lua_State *L) {
--  int i, n;
--  /* 1st argument must be a table (t) */
--  luaL_checktype(L, 1, LUA_TTABLE);
--  /* 2nd argument must be a function (f) */
--  luaL_checktype(L, 2, LUA_TFUNCTION);
--  n = luaL_len(L, 1); /* get size of table */
--  for (i = 1; i <= n; i++) {
--    lua_pushvalue(L, 2); /* push f */
--    lua_rawgeti(L, 1, i); /* push t[i] */
--    lua_call(L, 1, 1); /* call f(t[i]) */
--    lua_rawseti(L, 1, i); /* t[i] = result */
--  }
--  return 0; /* no results */
--}
--
--static int l_split (lua_State *L) {
--  const char *s = luaL_checkstring(L, 1); /* subject */
--  const char *sep = luaL_checkstring(L, 2); /* separator */
--  const char *e;
--  int i = 1;
--  lua_newtable(L); /* result table */
--  /* repeat for each separator */
--  while ((e = strchr(s, *sep)) != NULL) {
--    lua_pushlstring(L, s, e-s); /* push substring */
--    lua_rawseti(L, -2, i++); /* insert it in table */
--    s = e + 1; /* skip separator */
--  }
--  /* insert last substring */
--  lua_pushstring(L, s);
--  lua_rawseti(L, -2, i);
--  return 1; /* return the table */
--}
--
--static int str_upper (lua_State *L) {
--  size_t l;
--  size_t i;
--  luaL_Buffer b;
--  const char *s = luaL_checklstring(L, 1, &l);
--  char *p = luaL_buffinitsize(L, &b, l);
--  for (i = 0; i < l; i++)
--    p[i] = toupper(uchar(s[i]));
--  luaL_pushresultsize(&b, l);
--  return 1;
--}
--
--static int tconcat (lua_State *L) {
--  luaL_Buffer b;
--  int i, n;
--  luaL_checktype(L, 1, LUA_TTABLE);
--  n = luaL_len(L, 1);
--  luaL_buffinit(L, &b);
--  for (i = 1; i <= n; i++) {
--    lua_rawgeti(L, 1, i); /* get string from table */
--    luaL_addvalue(b); /* add it to the buffer */
--  }
--  luaL_pushresult(&b);
--  return 1;
--}
--
--
--int t_tuple (lua_State *L) {
--  int op = luaL_optint(L, 1, 0);
--  if (op == 0) { /* no arguments? */
--  int i;
--  /* push each valid upvalue onto the stack */
--  for (i = 1; !lua_isnone(L, lua_upvalueindex(i)); i++)
--    lua_pushvalue(L, lua_upvalueindex(i));
--    return i - 1; /* number of values in the stack */
--  }
--  else { /* get field 'op' */
--    luaL_argcheck(L, 0 < op, 1, "index out of range");
--    if (lua_isnone(L, lua_upvalueindex(op)))
--    return 0; /* no such field */
--    lua_pushvalue(L, lua_upvalueindex(op));
--    return 1;
--  }
--}
--int t_new (lua_State *L) {
--  lua_pushcclosure(L, t_tuple, lua_gettop(L));
--  return 1;
--}
--static const struct luaL_Reg tuplelib [] = {
--  {"new", t_new},
--  {NULL, NULL}
--};
--int luaopen_tuple (lua_State *L) {
--  luaL_newlib(L, tuplelib);
--  return 1;
--}