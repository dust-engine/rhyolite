extern crate proc_macro;
use std::{borrow::Borrow, cell::RefCell};

use quote::ToTokens;
use syn::{
    parse::{Parse, ParseStream},
    spanned::Spanned,
    token, ExprUnsafe,
};

struct ExprGpuAsync {
    pub stmts: Vec<syn::Stmt>,
}
impl Parse for ExprGpuAsync {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(ExprGpuAsync {
            stmts: syn::Block::parse_within(input)?,
        })
    }
}

#[proc_macro]
pub fn commands(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro_commands(input.into()).into()
}

struct ForkInput {
    forked_future: syn::Expr,
    number_of_forks: Option<(syn::Token![,], usize)>,
    comma: syn::Token![,],
    scope: syn::Block,
}
impl Parse for ForkInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let forked_future = input.parse()?;
        let comma: syn::Token![,] = input.parse()?;
        let number_of_forks = {
            let number: syn::LitInt = input.parse()?;
            let number = number.base10_parse::<usize>()?;
            Some(number)
        };
        let number_of_forks = if let Some(number_of_forks) = number_of_forks {
            Some((comma, number_of_forks))
        } else {
            None
        };

        Ok(ForkInput {
            forked_future,
            number_of_forks,
            comma: input.parse()?,
            scope: input.parse()?,
        })
    }
}

struct CommandsTransformState {
    current_import_id: usize,
    import_bindings: proc_macro2::TokenStream,
    import_drops: proc_macro2::TokenStream,
    import_retained_states: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,

    current_await_index: usize,
    await_bindings: proc_macro2::TokenStream,
    await_drops: proc_macro2::TokenStream,
    await_retained_states: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
}
impl Default for CommandsTransformState {
    fn default() -> Self {
        use proc_macro2::TokenStream;
        use syn::punctuated::Punctuated;
        Self {
            current_import_id: 0,
            import_bindings: TokenStream::new(),
            import_drops: TokenStream::new(),
            import_retained_states: Punctuated::new(),

            current_await_index: 0,
            await_bindings: TokenStream::new(),
            await_drops: TokenStream::new(),
            await_retained_states: Punctuated::new(),
        }
    }
}

use syn::{Block, Expr, Pat, Stmt};
impl CommandsTransformState {
    fn transform_block(&mut self, block: &Block) -> Block {
        Block {
            brace_token: block.brace_token.clone(),
            stmts: block
                .stmts
                .iter()
                .map(move |stmt| self.transform_stmt(stmt))
                .collect(),
        }
    }
    fn transform_pattern(&mut self, pat: &Pat) -> Pat {
        match pat {
            Pat::Box(pat) => Pat::Box(syn::PatBox {
                pat: Box::new(self.transform_pattern(&pat.pat)),
                ..pat.clone()
            }),
            Pat::Ident(ident) => Pat::Ident(syn::PatIdent {
                subpat: ident
                    .subpat
                    .as_ref()
                    .map(|(at, subpat)| (at.clone(), Box::new(self.transform_pattern(subpat)))),
                ..ident.clone()
            }),
            Pat::Lit(lit) => Pat::Lit(syn::PatLit {
                expr: Box::new(self.transform_expr(&lit.expr)),
                ..lit.clone()
            }),
            Pat::Or(clause) => Pat::Or(syn::PatOr {
                cases: clause
                    .cases
                    .iter()
                    .map(|pat| self.transform_pattern(pat))
                    .collect(),
                ..clause.clone()
            }),
            Pat::Range(range) => Pat::Range(syn::PatRange {
                lo: Box::new(self.transform_expr(&range.lo)),
                hi: Box::new(self.transform_expr(&range.hi)),
                ..range.clone()
            }),
            Pat::Reference(r) => Pat::Reference(syn::PatReference {
                pat: Box::new(self.transform_pattern(&r.pat)),
                ..r.clone()
            }),
            Pat::Slice(slice) => Pat::Slice(syn::PatSlice {
                elems: slice
                    .elems
                    .iter()
                    .map(|pat| self.transform_pattern(pat))
                    .collect(),
                ..slice.clone()
            }),
            Pat::Struct(s) => Pat::Struct(syn::PatStruct {
                fields: s
                    .fields
                    .iter()
                    .map(|f| syn::FieldPat {
                        pat: Box::new(self.transform_pattern(&f.pat)),
                        ..f.clone()
                    })
                    .collect(),
                ..s.clone()
            }),
            Pat::Tuple(tuple) => Pat::Tuple(syn::PatTuple {
                elems: tuple
                    .elems
                    .iter()
                    .map(|pat| self.transform_pattern(pat))
                    .collect(),
                ..tuple.clone()
            }),
            Pat::TupleStruct(tuple) => Pat::TupleStruct(syn::PatTupleStruct {
                pat: syn::PatTuple {
                    elems: tuple
                        .pat
                        .elems
                        .iter()
                        .map(|pat| self.transform_pattern(pat))
                        .collect(),
                    ..tuple.pat.clone()
                },
                ..tuple.clone()
            }),
            Pat::Type(ty) => Pat::Type(syn::PatType {
                pat: Box::new(self.transform_pattern(&ty.pat)),
                ..ty.clone()
            }),
            _ => pat.clone(),
        }
    }
    fn transform_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Array(arr) => Expr::Array(syn::ExprArray {
                elems: arr
                    .elems
                    .iter()
                    .map(|expr| self.transform_expr(expr))
                    .collect(),
                ..arr.clone()
            }),
            Expr::Assign(assign) => Expr::Assign(syn::ExprAssign {
                left: Box::new(self.transform_expr(&*assign.left)),
                right: Box::new(self.transform_expr(&*assign.right)),
                ..assign.clone()
            }),
            Expr::AssignOp(assign) => Expr::AssignOp(syn::ExprAssignOp {
                left: Box::new(self.transform_expr(&*assign.left)),
                right: Box::new(self.transform_expr(&*assign.right)),
                ..assign.clone()
            }),
            Expr::Binary(binary) => Expr::Binary(syn::ExprBinary {
                left: Box::new(self.transform_expr(&*binary.left)),
                right: Box::new(self.transform_expr(&*binary.right)),
                ..binary.clone()
            }),
            Expr::Block(block) => Expr::Block(syn::ExprBlock {
                block: self.transform_block(&block.block),
                ..block.clone()
            }),
            Expr::Call(call) => Expr::Call(syn::ExprCall {
                func: Box::new(self.transform_expr(&*call.func)),
                args: call
                    .args
                    .iter()
                    .map(|expr| self.transform_expr(expr))
                    .collect(),
                ..call.clone()
            }),
            Expr::Cast(cast) => Expr::Cast(syn::ExprCast {
                expr: Box::new(self.transform_expr(&*cast.expr)),
                ..cast.clone()
            }),
            Expr::Field(field) => Expr::Field(syn::ExprField {
                base: Box::new(self.transform_expr(&*field.base)),
                ..field.clone()
            }),
            Expr::ForLoop(l) => Expr::ForLoop(syn::ExprForLoop {
                pat: self.transform_pattern(&l.pat),
                expr: Box::new(self.transform_expr(&*l.expr)),
                body: self.transform_block(&l.body),
                ..l.clone()
            }),
            Expr::Group(group) => Expr::Group(syn::ExprGroup {
                expr: Box::new(self.transform_expr(&group.expr)),
                ..group.clone()
            }),
            Expr::If(if_stmt) => Expr::If(syn::ExprIf {
                cond: Box::new(self.transform_expr(&if_stmt.cond)),
                then_branch: self.transform_block(&if_stmt.then_branch),
                else_branch: if_stmt.else_branch.as_ref().map(|(else_token, expr)| {
                    (else_token.clone(), Box::new(self.transform_expr(&expr)))
                }),
                ..if_stmt.clone()
            }),
            Expr::Index(index_expr) => Expr::Index(syn::ExprIndex {
                expr: Box::new(self.transform_expr(&*index_expr.expr)),
                index: Box::new(self.transform_expr(&*index_expr.index)),
                ..index_expr.clone()
            }),
            Expr::Let(l) => Expr::Let(syn::ExprLet {
                pat: self.transform_pattern(&l.pat),
                expr: Box::new(self.transform_expr(&*l.expr)),
                ..l.clone()
            }),
            Expr::Loop(loop_stmt) => Expr::Loop(syn::ExprLoop {
                body: self.transform_block(&loop_stmt.body),
                ..loop_stmt.clone()
            }),
            Expr::Macro(m) => self.macro_transform(m),
            Expr::Match(match_expr) => Expr::Match(syn::ExprMatch {
                expr: Box::new(self.transform_expr(&*match_expr.expr)),
                arms: match_expr
                    .arms
                    .iter()
                    .map(|arm| syn::Arm {
                        pat: self.transform_pattern(&arm.pat),
                        guard: arm.guard.as_ref().map(|(guard_token, expr)| {
                            (guard_token.clone(), Box::new(self.transform_expr(&expr)))
                        }),
                        body: Box::new(self.transform_expr(&arm.body)),
                        ..arm.clone()
                    })
                    .collect(),
                ..match_expr.clone()
            }),
            Expr::MethodCall(call) => Expr::MethodCall(syn::ExprMethodCall {
                receiver: Box::new(self.transform_expr(&call.receiver)),
                args: call
                    .args
                    .iter()
                    .map(|expr| self.transform_expr(expr))
                    .collect(),
                ..call.clone()
            }),
            Expr::Paren(paren) => Expr::Paren(syn::ExprParen {
                expr: Box::new(self.transform_expr(&paren.expr)),
                ..paren.clone()
            }),
            Expr::Range(range) => Expr::Range(syn::ExprRange {
                from: range
                    .from
                    .as_ref()
                    .map(|f| Box::new(self.transform_expr(&f))),
                to: range.to.as_ref().map(|t| Box::new(self.transform_expr(&t))),
                ..range.clone()
            }),
            Expr::Reference(reference) => Expr::Reference(syn::ExprReference {
                expr: Box::new(self.transform_expr(&reference.expr)),
                ..reference.clone()
            }),
            Expr::Repeat(repeat) => Expr::Repeat(syn::ExprRepeat {
                expr: Box::new(self.transform_expr(&repeat.expr)),
                len: Box::new(self.transform_expr(&repeat.len)),
                ..repeat.clone()
            }),
            Expr::Return(ret) => {
                if let Some(expr) = self.return_transform(ret) {
                    expr
                } else {
                    Expr::Return(syn::ExprReturn {
                        expr: ret.expr.as_ref().map(|e| Box::new(self.transform_expr(&e))),
                        ..ret.clone()
                    })
                }
            }
            Expr::Struct(s) => Expr::Struct(syn::ExprStruct {
                fields: s
                    .fields
                    .iter()
                    .map(|f| syn::FieldValue {
                        member: f.member.clone(),
                        expr: self.transform_expr(&f.expr),
                        ..f.clone()
                    })
                    .collect(),
                ..s.clone()
            }),
            Expr::Try(s) => Expr::Try(syn::ExprTry {
                expr: Box::new(self.transform_expr(&s.expr)),
                ..s.clone()
            }),
            Expr::Tuple(tuple) => Expr::Tuple(syn::ExprTuple {
                elems: tuple
                    .elems
                    .iter()
                    .map(|expr| self.transform_expr(expr))
                    .collect(),
                ..tuple.clone()
            }),
            Expr::Type(type_expr) => Expr::Type(syn::ExprType {
                expr: Box::new(self.transform_expr(&type_expr.expr)),
                ..type_expr.clone()
            }),
            Expr::Unary(unary) => Expr::Unary(syn::ExprUnary {
                expr: Box::new(self.transform_expr(&unary.expr)),
                ..unary.clone()
            }),
            Expr::Unsafe(unsafe_stmt) => Expr::Unsafe(ExprUnsafe {
                block: self.transform_block(&unsafe_stmt.block),
                ..unsafe_stmt.clone()
            }),
            Expr::While(while_stmt) => Expr::While(syn::ExprWhile {
                cond: Box::new(self.transform_expr(&while_stmt.cond)),
                body: self.transform_block(&while_stmt.body),
                ..while_stmt.clone()
            }),
            Expr::Await(await_expr) => self.async_transform(await_expr),
            _ => expr.clone(),
        }
    }
    fn transform_stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Local(local) => Stmt::Local(syn::Local {
                pat: self.transform_pattern(&local.pat),
                init: local.init.as_ref().map(|(eq_token, expr)| {
                    (eq_token.clone(), Box::new(self.transform_expr(&expr)))
                }),
                ..local.clone()
            }),
            Stmt::Expr(expr) => Stmt::Expr(self.transform_expr(&expr)),
            Stmt::Semi(expr, semi) => Stmt::Semi(self.transform_expr(expr), semi.clone()),
            _ => stmt.clone(),
        }
    }

    fn import(
        &mut self,
        input_tokens: &proc_macro2::TokenStream,
        is_image: bool,
    ) -> proc_macro2::TokenStream {
        let global_res_variable_name =
            quote::format_ident!("__future_res_{}", self.current_import_id);
        self.current_import_id += 1;
        let output_tokens = if is_image {
            quote::quote! {{
                __fut_global_ctx.add_image(&mut #global_res_variable_name)
            }}
        } else {
            quote::quote! {{
                __fut_global_ctx.add_res(&mut #global_res_variable_name)
            }}
        };
        self.import_bindings.extend(quote::quote! {
            let mut #global_res_variable_name = #input_tokens;
        });
        self.import_drops.extend(quote::quote! {
            drop(#global_res_variable_name);
        });
        self.import_retained_states
            .push(syn::Expr::Verbatim(quote::quote! {
                #global_res_variable_name
            }));
        output_tokens
    }
    fn async_transform(&mut self, input: &syn::ExprAwait) -> syn::Expr {
        let global_future_variable_name =
            quote::format_ident!("__future_{}", self.current_await_index);
        self.current_await_index += 1;

        self.await_bindings.extend(quote::quote! {
            let mut #global_future_variable_name;
        });
        self.await_drops.extend(quote::quote! {
            drop(#global_future_variable_name);
        });
        self.await_retained_states
            .push(syn::Expr::Verbatim(quote::quote! {
                #global_future_variable_name
            }));

        let base = input.base.clone();

        // For each future, we first call init on it. This is a no-op for most futures, but for
        // (nested) block futures, this is going to move the future to its first yield point.
        // At that point it should yield the context of the first future.
        let tokens = quote::quote! {
            {
                let mut fut = #base;
                let mut fut_pinned = unsafe { std::pin::Pin::new_unchecked(&mut fut) };
                fut_pinned.as_mut().init(__fut_global_ctx);
                let mut ctx = Default::default();
                fut_pinned.as_mut().context(&mut ctx);
                yield ctx;
                loop {
                    match fut_pinned.as_mut().record(__fut_global_ctx) {
                        std::task::Poll::Ready((ret, retained_state)) => {
                            #global_future_variable_name = retained_state;
                            break ret
                        },
                        std::task::Poll::Pending => {
                            let mut ctx = Default::default();
                            fut_pinned.as_mut().context(&mut ctx);
                            yield ctx;
                        },
                    };
                }
                drop(fut);
            }
        };
        syn::parse2(tokens).unwrap()
    }
    fn macro_transform(&mut self, mac: &syn::ExprMacro) -> syn::Expr {
        let path = &mac.mac.path;
        if path.segments.len() != 1 {
            return syn::Expr::Macro(mac.clone());
        }
        match path.segments[0].ident.to_string().as_str() {
            "import" => syn::Expr::Verbatim(self.import(&mac.mac.tokens, false)),
            "import_image" => syn::Expr::Verbatim(self.import(&mac.mac.tokens, true)),
            "fork" => syn::Expr::Verbatim(self.fork_transform(&mac.mac.tokens)),
            _ => syn::Expr::Macro(mac.clone()),
        }
    }
    fn return_transform(&mut self, ret: &syn::ExprReturn) -> Option<syn::Expr> {
        // Transform each return statement into a yield, drops, and return.
        // We use RefCell on awaited_future_drops and import_drops so that they can be read while being modified.
        // This ensures that we won't drop uninitialized values.'
        // The executor will stop the execution as soon as it reaches the first `yield Complete` statement.
        // Drops are written between the yield and return, so these values are retained inside the generator
        // until the generator itself was dropped.
        // We drop the generator only after semaphore was signaled from within the queue.
        let returned_item = ret.expr.as_ref();
        let awaited_future_drops = &*self.await_drops.borrow();
        let awaited_future_retained_states = &*self.await_retained_states.borrow();
        let import_drops = &self.import_drops;
        let import_retained_states = &self.import_retained_states;
        let token_stream = quote::quote!(
            {
                return (#returned_item, ((#awaited_future_retained_states), (#import_retained_states)));
            }
        );
        let block = syn::parse2::<syn::ExprBlock>(token_stream).unwrap();
        Some(syn::Expr::Block(block))
    }

    fn fork_transform(&mut self, input: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        let ForkInput {
            forked_future,
            number_of_forks,
            comma: _,
            scope,
        } = match syn::parse2::<ForkInput>(input.clone()) {
            Ok(input) => input,
            Err(err) => return err.to_compile_error(),
        };
        let number_of_forks = number_of_forks.map(|(_, number)| number).unwrap_or(2);
        let ret = syn::Expr::Tuple(syn::ExprTuple {
            attrs: Vec::new(),
            paren_token: Default::default(),
            elems: (0..number_of_forks)
                .map(|_| {
                    syn::Expr::Verbatim(quote::quote! {
                        GPUCommandForked::new(&forked_future_inner)
                    })
                })
                .collect(),
        });
        // TODO: transform scope before feeding it in.
        let scope = syn::Block {
            brace_token: scope.brace_token.clone(),
            stmts: scope
                .stmts
                .iter()
                .map(|stmt| self.transform_stmt(stmt))
                .collect(),
        };
        quote::quote! {{
            let mut forked_future = GPUCommandForkedInner::wrap(#forked_future);
            let mut pinned = unsafe{std::pin::Pin::new_unchecked(&mut forked_future)};
            pinned.as_mut().unwrap_pinned().init(__fut_global_ctx);
            let forked_future_inner = GPUCommandForkedInner::new(pinned);
            let #forked_future = #ret;
            #scope
        }}
    }
}

fn proc_macro_commands(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<ExprGpuAsync>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };
    let mut state = CommandsTransformState::default();

    let mut inner_closure_stmts: Vec<_> = input
        .stmts
        .iter()
        .map(|stmt| state.transform_stmt(stmt))
        .collect();

    // Transform the final return
    if let Some(last) = inner_closure_stmts.last_mut() {
        let import_retained_states = &*state.import_retained_states.borrow();
        let awaited_future_retained_states = &*state.await_retained_states.borrow();

        if let syn::Stmt::Expr(expr) = last {
            let token_stream = quote::quote!(
                {
                    return (#expr, ((#awaited_future_retained_states), (#import_retained_states)));
                }
            );
            let block = syn::parse2::<syn::ExprBlock>(token_stream).unwrap();
            *expr = syn::Expr::Block(block);
        } else {
            let token_stream = quote::quote!(
                {
                    return ((), ((#awaited_future_retained_states), (#import_retained_states)));
                }
            );
            let block = syn::parse2::<syn::ExprBlock>(token_stream).unwrap();
            inner_closure_stmts.push(syn::Stmt::Expr(syn::Expr::Block(block)))
        }
    }

    let import_bindings = state.import_bindings;
    let awaited_future_bindings = state.await_bindings;
    quote::quote! {
        async_ash::future::GPUCommandBlock::new(static |__fut_global_ctx: *mut ::async_ash::future::GlobalContext| {
            let __fut_global_ctx: &mut ::async_ash::future::GlobalContext = unsafe{&mut *__fut_global_ctx};
            #import_bindings
            #awaited_future_bindings
            #(#inner_closure_stmts)*
        })
    }
}

struct MacroJoin {
    pub exprs: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
}
impl Parse for MacroJoin {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(MacroJoin {
            exprs: syn::punctuated::Punctuated::parse_separated_nonempty(input)?,
        })
    }
}

#[proc_macro]
pub fn join(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro_join(input.into()).into()
}

fn proc_macro_join(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<MacroJoin>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };
    if input.exprs.len() == 0 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "Expects at least one argument",
        )
        .to_compile_error();
    }
    if input.exprs.len() == 1 {
        return input.exprs[0].clone().into_token_stream();
    }

    let mut token_stream = proc_macro2::TokenStream::new();
    token_stream.extend(input.exprs[0].clone().into_token_stream());

    // a.join(b).join(c)...
    for item in input.exprs.iter().skip(1) {
        token_stream.extend(quote::quote! {.join(#item)}.into_iter());
    }

    let num_expressions = input.exprs.len();

    // __join_0, __join_1, __join_2, ...
    let output_expression = (0..num_expressions).map(|i| quote::format_ident!("__join_{}", i));

    let input_expression = {
        use proc_macro2::{Delimiter, Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
        let mut t = Some(TokenTree::Group(Group::new(Delimiter::Parenthesis, {
            let mut t = TokenStream::new();
            t.extend(Some(TokenTree::Ident(Ident::new(
                "__join_0",
                Span::call_site(),
            ))));
            t.extend(Some(TokenTree::Punct(Punct::new(',', Spacing::Alone))));
            t.extend(Some(TokenTree::Ident(Ident::new(
                "__join_1",
                Span::call_site(),
            ))));
            t
        })));
        (2..num_expressions).for_each(|i| {
            let prev = t.take().unwrap().into_token_stream();
            t = Some(TokenTree::Group(Group::new(Delimiter::Parenthesis, {
                let mut a = TokenStream::new();
                a.extend(Some(prev));
                a.extend(Some(TokenTree::Punct(Punct::new(',', Spacing::Alone))));
                a.extend(Some(TokenTree::Ident(quote::format_ident!("__join_{}", i))));
                a
            })));
        });
        t
    };
    token_stream
        .extend(quote::quote! {.map(|#input_expression| (#(#output_expression),*))}.into_iter());
    token_stream
}
